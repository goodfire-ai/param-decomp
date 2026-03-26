"""
Runs a Jupyter Server for the Scribe MCP server to connect to.
"""

import asyncio
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import nbformat
from jupyter_server.serverapp import ServerApp
from traitlets import Int, Unicode

from scribe.notebook._notebook_server_utils import clean_notebook_for_save
from scribe.notebook._types import AsyncStarted, CellOutput, CellResult, ExecuteResponse, ReadResponse

from . import notebook_server_handlers as _handlers


# Request/Response models as simple dicts for Tornado handlers
@dataclass
class ScribeNotebookSession:
    """Container for all session-related data."""

    session_id: str
    kernel_id: str
    jupyter_session_id: str
    notebook_path: Path
    script_path: Path
    display_name: str
    execution_count: int = 0
    last_activity: datetime | None = None


@dataclass
class _Execution:
    """Tracks a background async execution. Only stores task handle — outputs live in notebook cells."""

    status: str = "running"
    task: asyncio.Task | None = None


class ScribeServerApp(ServerApp):
    """Jupyter Server app with Scribe customizations."""

    notebooks_dir = Unicode(
        "notebooks",
        config=True,
        help="Directory for saving notebooks. Supports ~ expansion and environment variables.",
    )

    auto_shutdown_minutes = Int(
        60, config=True, help="Minutes before auto-shutdown of idle kernels"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Single source of truth for all session data
        # Maps session ID to session instance
        self.sessions: dict[str, ScribeNotebookSession] = {}

        # Track last activity time
        self.last_activity_time = datetime.now()

        # Set up auto-shutdown timer if enabled
        if self.auto_shutdown_minutes > 0:
            from tornado.ioloop import PeriodicCallback

            self.shutdown_check_callback = PeriodicCallback(
                self.check_auto_shutdown,
                60000,  # Check every minute
            )

        # Note: notebooks_path will be set up in initialize() after config is parsed
        self.notebooks_path = None

        # Debounce state for notebook writes (prevents IDE thrashing)
        self._notebook_buffers: dict[str, nbformat.NotebookNode] = {}  # session_id -> in-memory notebook
        self._last_write_time: dict[str, float] = {}  # session_id -> timestamp of last disk write
        self._pending_flush_handles: dict[str, object] = {}  # session_id -> scheduled flush handle
        self._write_debounce_seconds: float = 1.5  # minimum time between writes

        # Async cell execution tracking
        self._async_executions: dict[str, _Execution] = {}  # session_id -> running execution

    def initialize(self, argv=None):
        """Initialize the server after parsing configuration."""
        # Call parent initialization first to parse config
        super().initialize(argv)

        # Now set up notebooks directory with the parsed configuration
        self.notebooks_path = self._setup_notebooks_directory()

    def _setup_notebooks_directory(self) -> Path:
        """Set up and validate the notebooks directory with enhanced path handling."""
        try:
            # Expand user home directory (~) and environment variables
            expanded_path = os.path.expanduser(os.path.expandvars(self.notebooks_dir))

            # Convert to Path object
            notebooks_path = Path(expanded_path)

            # Make it absolute if it's relative
            if not notebooks_path.is_absolute():
                notebooks_path = Path.cwd() / notebooks_path

            # Resolve any relative components (like .. or .)
            notebooks_path = notebooks_path.resolve()

            # Check if path exists and is a file (not allowed)
            if notebooks_path.exists() and notebooks_path.is_file():
                raise ValueError(
                    f"Notebooks directory path '{notebooks_path}' exists but is a file, not a directory"
                )

            # Create directory if it doesn't exist
            notebooks_path.mkdir(parents=True, exist_ok=True)

            # Verify we can write to the directory
            test_file = notebooks_path / ".scribe_write_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except PermissionError:
                raise ValueError(
                    f"No write permission for notebooks directory: {notebooks_path}"
                )

            print(f"📁 Notebooks directory: {notebooks_path}")
            return notebooks_path

        except Exception as e:
            error_msg = (
                f"Failed to set up notebooks directory '{self.notebooks_dir}': {str(e)}"
            )
            print(f"❌ {error_msg}")
            raise ValueError(error_msg) from e

    def init_webapp(self):
        """Add our custom handlers to the web app."""
        super().init_webapp()

        # Add Scribe API handlers
        host_pattern = ".*$"
        base_url = self.base_url

        handlers = [
            (f"{base_url}api/scribe/start", _handlers.StartSessionHandler),
            (f"{base_url}api/scribe/restart", _handlers.RestartKernelHandler),
            (f"{base_url}api/scribe/shutdown", _handlers.ShutdownSessionHandler),
            (f"{base_url}api/scribe/sync", _handlers.SyncFileCellsHandler),
            (f"{base_url}api/scribe/execute", _handlers.ExecuteHandler),
            (f"{base_url}api/scribe/execute-async", _handlers.ExecuteAsyncHandler),
            (f"{base_url}api/scribe/read", _handlers.ReadHandler),
            (f"{base_url}api/scribe/cancel", _handlers.CancelHandler),
            (f"{base_url}api/scribe/health", _handlers.HealthCheckHandler),
            (f"{base_url}tree", _handlers.TreeHandler),
        ]

        self.web_app.add_handlers(host_pattern, handlers)

        # Start the auto-shutdown timer after webapp is initialized
        if hasattr(self, "shutdown_check_callback"):
            self.shutdown_check_callback.start()

    async def start_session(self, experiment_name: str | None = None, notebook_dir: str | None = None):
        """Start a new scribe session — creates notebook, kernel, and session."""
        if self.notebooks_path is None:
            self.notebooks_path = self._setup_notebooks_directory()

        self.update_activity()

        target_dir = Path(notebook_dir) if notebook_dir else self.notebooks_path
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        base_name = f"{timestamp}_{experiment_name or 'Notebook'}"

        nb_path = target_dir / f"{base_name}.ipynb"
        counter = 1
        while nb_path.exists():
            nb_path = target_dir / f"{base_name}_{counter}.ipynb"
            counter += 1

        nb = nbformat.v4.new_notebook()
        nb.metadata.update({
            "kernelspec": {"display_name": f"Scribe: {base_name}", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        })
        with open(nb_path, "w") as f:
            nbformat.write(clean_notebook_for_save(nb), f)

        try:
            relative_path = nb_path.relative_to(Path.cwd())
        except ValueError:
            relative_path = nb_path

        kernel_id = await self.kernel_manager.start_kernel()
        sm = self.web_app.settings["session_manager"]
        jupyter_session = await sm.create_session(
            path=str(relative_path), type="notebook",
            name=nb_path.name, kernel_id=kernel_id,
        )

        session_id = str(uuid.uuid4())
        self.sessions[session_id] = ScribeNotebookSession(
            session_id=session_id, kernel_id=kernel_id,
            jupyter_session_id=jupyter_session["id"], notebook_path=nb_path,
            script_path=nb_path.with_suffix(".py"), display_name=f"Scribe: {base_name}",
            last_activity=datetime.now(),
        )

        return {
            "session_id": session_id, "kernel_id": kernel_id,
            "notebook_path": str(nb_path), "script_path": str(nb_path.with_suffix(".py")),
            "kernel_display_name": f"Scribe: {base_name}",
        }

    async def _add_pending_cell(self, session_id: str, code: str) -> int:
        """
        Add a code cell with pending execution status.
        Used to immediately add a code cell to the notebook file before execution
        begins, giving users visual feedback that something is happening.
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Use buffered notebook if available, otherwise read from disk
        nb = self._notebook_buffers.get(session_id)
        if nb is None:
            with open(session.notebook_path) as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        # Get next execution count
        session.execution_count += 1
        execution_count = session.execution_count

        # Create new cell with pending status
        cell = nbformat.v4.new_code_cell(
            source=code,
            outputs=[],
            execution_count=execution_count,
            metadata={"execution_status": "pending"},
        )

        # Append cell
        nb.cells.append(cell)
        cell_index = len(nb.cells) - 1

        # Store in buffer and write immediately (first write is always immediate)
        self._notebook_buffers[session_id] = nb
        self._flush_notebook_to_disk(session_id)

        return cell_index

    def _flush_notebook_to_disk(self, session_id: str) -> None:
        """Write the buffered notebook to disk and update timing."""
        session = self.sessions.get(session_id)
        nb = self._notebook_buffers.get(session_id)
        if not session or not nb:
            return

        with open(session.notebook_path, "w") as f:
            nbformat.write(clean_notebook_for_save(nb), f)
        self._last_write_time[session_id] = time.time()

        # Clear any pending flush since we just wrote
        if session_id in self._pending_flush_handles:
            handle = self._pending_flush_handles.pop(session_id)
            from tornado.ioloop import IOLoop
            IOLoop.current().remove_timeout(handle)

    def _schedule_flush(self, session_id: str, delay: float) -> None:
        """Schedule a delayed flush to disk."""
        # Cancel any existing scheduled flush
        if session_id in self._pending_flush_handles:
            from tornado.ioloop import IOLoop
            handle = self._pending_flush_handles.pop(session_id)
            IOLoop.current().remove_timeout(handle)

        # Schedule new flush
        from tornado.ioloop import IOLoop
        handle = IOLoop.current().call_later(
            delay,
            lambda: self._flush_notebook_to_disk(session_id)
        )
        self._pending_flush_handles[session_id] = handle

    def _throttled_write(self, session_id: str) -> None:
        """Write to disk respecting the debounce interval."""
        now = time.time()
        last_write = self._last_write_time.get(session_id, 0)
        time_since_last = now - last_write

        if time_since_last >= self._write_debounce_seconds:
            # Enough time has passed, write immediately
            self._flush_notebook_to_disk(session_id)
        else:
            # Schedule a write for when the debounce period expires
            delay = self._write_debounce_seconds - time_since_last
            if session_id not in self._pending_flush_handles:
                self._schedule_flush(session_id, delay)

    async def _update_cell_output(
        self, session_id: str, cell_index: int, output: dict, status: str
    ):
        """Update a cell with a new output (uses throttled writes)."""
        session = self.sessions.get(session_id)
        if not session:
            return

        # Use buffered notebook if available, otherwise read from disk
        nb = self._notebook_buffers.get(session_id)
        if nb is None:
            with open(session.notebook_path) as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
            self._notebook_buffers[session_id] = nb

        if cell_index >= len(nb.cells):
            return

        cell = nb.cells[cell_index]

        # Convert output dict to nbformat output
        if output["output_type"] == "stream":
            cell_output = nbformat.v4.new_output(
                output_type="stream", name=output["name"], text=output["text"]
            )
        elif output["output_type"] == "execute_result":
            cell_output = nbformat.v4.new_output(
                output_type="execute_result",
                data=output["data"],
                metadata=output.get("metadata", {}),
                execution_count=output.get("execution_count"),
            )
        elif output["output_type"] == "display_data":
            cell_output = nbformat.v4.new_output(
                output_type="display_data",
                data=output["data"],
                metadata=output.get("metadata", {}),
            )
        elif output["output_type"] == "error":
            cell_output = nbformat.v4.new_output(
                output_type="error",
                ename=output["ename"],
                evalue=output["evalue"],
                traceback=output["traceback"],
            )
        else:
            return

        # Append output to in-memory buffer
        cell.outputs.append(cell_output)

        # Update status
        cell.metadata["execution_status"] = status

        # Throttled write to disk
        self._throttled_write(session_id)

    async def _execute_and_stream(self, session_id: str, code: str):
        """Execute code and yield outputs as they arrive."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        kernel_id = session.kernel_id

        # Create a client to execute code
        kernel = self.kernel_manager.get_kernel(kernel_id)
        client = kernel.client()
        client.start_channels()

        try:
            msg_id = client.execute(code)

            while True:
                msg = await client._async_get_iopub_msg(timeout=300)

                if msg["parent_header"].get("msg_id") != msg_id:
                    continue

                msg_type = msg["msg_type"]
                content = msg["content"]

                match msg_type:
                    case "execute_input":
                        session.execution_count = content["execution_count"]
                    case "stream":
                        yield {"output_type": "stream", "name": content["name"], "text": content["text"]}
                    case "execute_result":
                        yield {"output_type": "execute_result", "data": content["data"],
                               "metadata": content.get("metadata", {}), "execution_count": content["execution_count"]}
                    case "display_data":
                        yield {"output_type": "display_data", "data": content["data"],
                               "metadata": content.get("metadata", {})}
                    case "error":
                        yield {"output_type": "error", "ename": content["ename"],
                               "evalue": content["evalue"], "traceback": content["traceback"]}
                    case "status" if content["execution_state"] == "idle":
                        break

        finally:
            client.stop_channels()

    async def _update_cell_status(self, session_id: str, cell_index: int, status: str):
        """Update just the status of a cell (forces immediate flush)."""
        session = self.sessions.get(session_id)
        if not session:
            return

        # Use buffered notebook if available, otherwise read from disk
        nb = self._notebook_buffers.get(session_id)
        if nb is None:
            with open(session.notebook_path) as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
            self._notebook_buffers[session_id] = nb

        if cell_index < len(nb.cells):
            nb.cells[cell_index].metadata["execution_status"] = status

            # Force immediate flush since execution is complete
            self._flush_notebook_to_disk(session_id)

    def _resolve_tags(self, session_id: str, tags: list[str] | None) -> list[tuple[str, int]]:
        """Resolve tags to (tag, cell_index) pairs. Validates tags exist and no async conflict."""
        self.update_activity()
        self.update_session_activity(session_id)

        session = self.sessions.get(session_id)
        assert session, f"Session {session_id} not found"

        nb = self._notebook_buffers.get(session_id)
        if nb is None:
            with open(session.notebook_path) as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
            self._notebook_buffers[session_id] = nb

        tagged_cells = [
            (cell.metadata["scribe_tag"], i)
            for i, cell in enumerate(nb.cells)
            if cell.metadata.get("scribe_tag") and cell.cell_type == "code"
            and (tags is None or cell.metadata["scribe_tag"] in tags)
        ]

        if tags is not None:
            found = {t for t, _ in tagged_cells}
            missing = set(tags) - found
            assert not missing, f"Tags not found in notebook: {missing}"

        assert tagged_cells, "No matching code cells to execute"

        existing = self._async_executions.get(session_id)
        if existing and existing.task and not existing.task.done():
            raise ValueError("Session already has a running execution. Cancel it first.")

        return tagged_cells

    async def execute(self, session_id: str, tags: list[str] | None, timeout_per_cell: int = 300) -> ExecuteResponse:
        """Execute notebook cells by tag (blocking)."""
        tagged_cells = self._resolve_tags(session_id, tags)
        return await self._execute_cell_loop(session_id, tagged_cells, timeout_per_cell)

    async def execute_async(self, session_id: str, tags: list[str] | None) -> AsyncStarted:
        """Start executing notebook cells in the background (non-blocking)."""
        tagged_cells = self._resolve_tags(session_id, tags)
        async_exec = _Execution()
        self._async_executions[session_id] = async_exec
        async_exec.task = asyncio.create_task(
            self._execute_cell_loop_background(session_id, tagged_cells, 300, async_exec)
        )
        return AsyncStarted(n_cells=len(tagged_cells))

    async def _execute_cell_loop(
        self, session_id: str, tagged_cells: list[tuple[str, int]], timeout_per_cell: int
    ) -> ExecuteResponse:
        """Execute cells sequentially. Stops on first error."""
        session = self.sessions.get(session_id)
        assert session

        results: list[CellResult] = []
        for tag, cell_index in tagged_cells:
            nb = self._notebook_buffers.get(session_id)
            assert nb

            cell = nb.cells[cell_index]
            code = cell.source
            cell.outputs = []
            session.execution_count += 1
            cell.execution_count = session.execution_count
            cell.metadata["execution_status"] = "running"
            self._flush_notebook_to_disk(session_id)

            has_error = False
            try:
                async def _run_cell():
                    nonlocal has_error
                    async for output in self._execute_and_stream(session_id, code):
                        await self._update_cell_output(session_id, cell_index, output, "running")
                        if output.get("output_type") == "error":
                            has_error = True

                await asyncio.wait_for(_run_cell(), timeout=timeout_per_cell)

                status = "error" if has_error else "ok"
                await self._update_cell_status(session_id, cell_index, "error" if has_error else "complete")
                results.append(CellResult(tag=tag, status=status))
                if has_error:
                    break

            except asyncio.TimeoutError:
                await self.interrupt_kernel(session_id)
                await self._update_cell_status(session_id, cell_index, "error")
                results.append(CellResult(tag=tag, status="timeout"))
                break

            except Exception as e:
                await self._update_cell_output(session_id, cell_index, {
                    "output_type": "error", "ename": type(e).__name__,
                    "evalue": str(e), "traceback": [str(e)],
                }, "error")
                results.append(CellResult(tag=tag, status="error"))
                break

        return ExecuteResponse(cells=results)

    async def _execute_cell_loop_background(
        self, session_id: str, tagged_cells: list[tuple[str, int]],
        timeout_per_cell: int, async_exec: "_Execution",
    ):
        try:
            response = await self._execute_cell_loop(session_id, tagged_cells, timeout_per_cell)
            async_exec.status = "error" if any(c.status != "ok" for c in response.cells) else "complete"
        except Exception:
            async_exec.status = "error"

    def read_cells(self, session_id: str, cell_tag: str | None = None) -> ReadResponse:
        """Read cell data from the notebook by scribe_tag metadata."""
        session = self.sessions.get(session_id)
        assert session, f"Session {session_id} not found"

        nb = self._notebook_buffers.get(session_id)
        if nb is None:
            with open(session.notebook_path) as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

        async_exec = self._async_executions.get(session_id)
        async_status = async_exec.status if async_exec else "idle"

        cells: list[CellOutput] = []
        for cell in nb.cells:
            tag = cell.metadata.get("scribe_tag")
            if tag is None or (cell_tag is not None and tag != cell_tag):
                continue
            cells.append(CellOutput(
                tag=tag,
                cell_type=cell.cell_type,
                status=cell.metadata.get("execution_status", ""),
                outputs=list(cell.outputs) if cell.cell_type == "code" else [],
            ))

        if cell_tag and not cells:
            raise ValueError(f"Cell tag '{cell_tag}' not found in notebook")

        return ReadResponse(async_status=async_status, cells=cells)

    async def cancel_async_execution(self, session_id: str) -> dict:
        """Cancel a running async execution."""
        async_exec = self._async_executions.get(session_id)
        if async_exec is None or async_exec.status != "running":
            return {"status": "idle", "message": "No running execution to cancel"}

        if async_exec.task and not async_exec.task.done():
            async_exec.task.cancel()

        await self.interrupt_kernel(session_id)
        async_exec.status = "cancelled"

        return {"status": "cancelled"}

    async def sync_file_cells(
        self, session_id: str, cells: list[tuple[str, str, str]]
    ) -> dict[str, int]:
        """Sync named cells from a script file into the notebook.

        The notebook's tagged cells are rebuilt to mirror the file exactly:
        - Tagged cells appear in file order as a contiguous block
        - Changed cells have their source updated (outputs cleared for code cells)
        - New cells are inserted at the correct position
        - Orphaned tagged cells (tag no longer in file) are removed
        - Non-tagged cells are preserved before the block

        Args:
            session_id: The session ID
            cells: List of (tag, cell_type, source) tuples from the script file.
                   cell_type is "code" or "markdown".

        Returns:
            Dict mapping tag → notebook cell index
        """
        self.update_activity()
        self.update_session_activity(session_id)

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        nb = self._notebook_buffers.get(session_id)
        if nb is None:
            with open(session.notebook_path) as f:
                nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
            self._notebook_buffers[session_id] = nb

        # Partition existing cells and index tagged ones for reuse
        existing_by_tag: dict[str, nbformat.NotebookNode] = {}
        non_tagged: list[nbformat.NotebookNode] = []
        for cell in nb.cells:
            tag = cell.metadata.get("scribe_tag")
            if tag is not None:
                existing_by_tag[tag] = cell
            else:
                non_tagged.append(cell)

        # Build new tagged cell list in file order, tracking what changed
        dirty = False
        new_tagged: list[nbformat.NotebookNode] = []
        tag_to_nb_index: dict[str, int] = {}
        for tag, cell_type, source in cells:
            if tag in existing_by_tag:
                cell = existing_by_tag[tag]
                if cell.source != source:
                    cell.source = source
                    if cell.cell_type == "code":
                        cell.outputs = []
                        cell.metadata["execution_status"] = "stale"
                    dirty = True
                new_tagged.append(cell)
            else:
                if cell_type == "markdown":
                    new_cell = nbformat.v4.new_markdown_cell(source=source)
                else:
                    new_cell = nbformat.v4.new_code_cell(source=source)
                new_cell.metadata["scribe_tag"] = tag
                new_tagged.append(new_cell)
                dirty = True
            tag_to_nb_index[tag] = len(non_tagged) + len(new_tagged) - 1

        # Detect orphan removal as a change
        if set(existing_by_tag) - {tag for tag, _, _ in cells}:
            dirty = True

        # Rebuild: non-tagged cells, then tagged cells in file order
        nb.cells = non_tagged + new_tagged

        if dirty:
            self._notebook_buffers[session_id] = nb
            self._flush_notebook_to_disk(session_id)

        return tag_to_nb_index

    async def shutdown_session(self, session_id: str):
        """Shutdown a session and its kernel."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Flush any pending writes before shutdown
        if session_id in self._notebook_buffers:
            self._flush_notebook_to_disk(session_id)

        # Clean up debounce state
        self._notebook_buffers.pop(session_id, None)
        self._last_write_time.pop(session_id, None)
        if session_id in self._pending_flush_handles:
            from tornado.ioloop import IOLoop
            handle = self._pending_flush_handles.pop(session_id)
            IOLoop.current().remove_timeout(handle)

        # Delete the Jupyter session (this also shuts down the kernel)
        sm = self.web_app.settings["session_manager"]
        await sm.delete_session(session.jupyter_session_id)

        # Clean up our session tracking
        del self.sessions[session_id]

    async def restart_kernel(self, session_id: str) -> dict:
        """Restart a session's kernel, clearing all state but keeping the notebook."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Cancel any async execution
        async_exec = self._async_executions.get(session_id)
        if async_exec and async_exec.task and not async_exec.task.done():
            async_exec.task.cancel()
        self._async_executions.pop(session_id, None)

        # Restart the kernel (clears all state)
        await self.kernel_manager.restart_kernel(session.kernel_id)

        # Reset execution count
        session.execution_count = 0

        return {
            "session_id": session_id,
            "notebook_path": str(session.notebook_path),
            "status": "restarted",
        }

    async def interrupt_kernel(self, session_id: str) -> bool:
        """Interrupt the kernel for a session.

        Args:
            session_id: The session ID whose kernel to interrupt

        Returns:
            True if interrupt was sent successfully
        """
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        try:
            kernel = self.kernel_manager.get_kernel(session.kernel_id)
            await kernel.interrupt_kernel()
            return True
        except Exception as e:
            print(f"Failed to interrupt kernel for session {session_id}: {e}")
            return False

    def update_activity(self):
        """Update the last activity timestamp for entire ServerApp instance."""
        self.last_activity_time = datetime.now()

    def update_session_activity(self, session_id: str):
        """Update the last activity timestamp for a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.now()

    async def check_auto_shutdown(self):
        """Check if server should auto-shutdown due to inactivity."""
        current_time = datetime.now()

        # First, clean up inactive sessions
        inactive_sessions = []
        for session_id, session in self.sessions.items():
            session_idle_minutes = (
                current_time - session.last_activity
            ).total_seconds() / 60
            if session_idle_minutes >= self.auto_shutdown_minutes:
                inactive_sessions.append(session_id)

        # Remove inactive sessions
        for session_id in inactive_sessions:
            print(
                f"🧹 Cleaning up inactive session {session_id} (idle for {self.auto_shutdown_minutes}+ minutes)"
            )
            try:
                await self.shutdown_session(session_id)
            except Exception as e:
                print(f"   Error shutting down session {session_id}: {e}")
                # Still remove from tracking even if shutdown fails
                if session_id in self.sessions:
                    del self.sessions[session_id]

        # Now check if we should shutdown the server
        if not self.sessions:  # No active sessions remaining
            idle_minutes = (current_time - self.last_activity_time).total_seconds() / 60

            if idle_minutes >= self.auto_shutdown_minutes:
                print(
                    f"\n⏰ Auto-shutdown: Server idle for {int(idle_minutes)} minutes"
                )
                print("   Shutting down...")

                # Stop the periodic callback
                if hasattr(self, "shutdown_check_callback"):
                    self.shutdown_check_callback.stop()

                # Gracefully stop the server
                self.stop()

    def cleanup(self):
        """Clean up resources when server is shutting down."""
        # Call parent cleanup
        super().cleanup()


if __name__ == "__main__":
    """Entry point for running the Scribe server as a script."""
    import sys

    # Create and configure the server app
    app = ScribeServerApp()

    # Parse command line arguments
    if len(sys.argv) > 1:
        app.initialize(sys.argv[1:])
    else:
        app.initialize()

    # Start the server
    app.start()

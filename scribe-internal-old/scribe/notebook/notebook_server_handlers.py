"""Tornado HTTP handlers for the Scribe Jupyter Server API."""

import json

from jupyter_server.base.handlers import APIHandler
from tornado.web import authenticated


class ScribeAPIHandler(APIHandler):
    @property
    def scribe_app(self):
        return self.settings["serverapp"]

    def _handle_error(self, e: Exception):
        status = 400 if isinstance(e, (ValueError, AssertionError, KeyError)) else 500
        self.set_status(status)
        self.finish(json.dumps({"error": str(e)}))

    def _require_json(self, *required_keys: str) -> dict:
        data = self.get_json_body()
        assert data, "No JSON body provided"
        for key in required_keys:
            assert key in data, f"{key} is required"
        return data


class StartSessionHandler(ScribeAPIHandler):
    @authenticated
    async def post(self):
        try:
            data = self.get_json_body() or {}
            result = await self.scribe_app.start_session(
                experiment_name=data.get("experiment_name"),
                notebook_dir=data.get("notebook_dir"),
            )
            result["server_url"] = f"http://localhost:{self.scribe_app.port}"
            result["token"] = self.scribe_app.token
            result["vscode_url"] = f"http://localhost:{self.scribe_app.port}/?token={self.scribe_app.token}"
            result["kernel_name"] = result.pop("kernel_display_name", "")
            result["status"] = "started"
            self.finish(json.dumps(result))
        except Exception as e:
            self._handle_error(e)


class RestartKernelHandler(ScribeAPIHandler):
    @authenticated
    async def post(self):
        try:
            data = self._require_json("session_id")
            result = await self.scribe_app.restart_kernel(data["session_id"])
            self.finish(json.dumps(result))
        except Exception as e:
            self._handle_error(e)


class ShutdownSessionHandler(ScribeAPIHandler):
    @authenticated
    async def post(self):
        try:
            data = self._require_json("session_id")
            await self.scribe_app.shutdown_session(data["session_id"])
            self.finish(json.dumps({"status": "shutdown"}))
        except Exception as e:
            self._handle_error(e)


class SyncFileCellsHandler(ScribeAPIHandler):
    @authenticated
    async def post(self):
        try:
            data = self._require_json("session_id", "cells")
            tag_to_index = await self.scribe_app.sync_file_cells(data["session_id"], data["cells"])
            self.finish(json.dumps({"session_id": data["session_id"], "tag_to_index": tag_to_index}))
        except Exception as e:
            self._handle_error(e)


class ExecuteHandler(ScribeAPIHandler):
    @authenticated
    async def post(self):
        try:
            data = self._require_json("session_id")
            result = await self.scribe_app.execute(
                session_id=data["session_id"],
                tags=data.get("tags"),
                timeout_per_cell=data.get("timeout_per_cell", 300),
            )
            self.finish(result.model_dump_json())
        except Exception as e:
            self._handle_error(e)


class ExecuteAsyncHandler(ScribeAPIHandler):
    @authenticated
    async def post(self):
        try:
            data = self._require_json("session_id")
            result = await self.scribe_app.execute_async(
                session_id=data["session_id"],
                tags=data.get("tags"),
            )
            self.finish(result.model_dump_json())
        except Exception as e:
            self._handle_error(e)


class ReadHandler(ScribeAPIHandler):
    @authenticated
    async def post(self):
        try:
            data = self._require_json("session_id")
            result = self.scribe_app.read_cells(
                session_id=data["session_id"],
                cell_tag=data.get("cell_tag"),
            )
            self.finish(result.model_dump_json())
        except Exception as e:
            self._handle_error(e)


class CancelHandler(ScribeAPIHandler):
    @authenticated
    async def post(self):
        try:
            data = self._require_json("session_id")
            result = await self.scribe_app.cancel_async_execution(data["session_id"])
            self.finish(json.dumps(result))
        except Exception as e:
            self._handle_error(e)


class HealthCheckHandler(ScribeAPIHandler):
    async def get(self):
        self.finish(json.dumps({
            "status": "healthy",
            "server": "Scribe with Jupyter Server",
            "jupyter_url": f"http://localhost:{self.scribe_app.port}/?token={self.scribe_app.token}",
            "active_sessions": len(self.scribe_app.sessions),
            "notebooks_dir": str(self.scribe_app.notebooks_path.absolute()),
        }))


class TreeHandler(APIHandler):
    async def get(self):
        self.set_header("Content-Type", "text/html")
        self.finish("<h1>Scribe Jupyter Server</h1><p>Server is running.</p>")

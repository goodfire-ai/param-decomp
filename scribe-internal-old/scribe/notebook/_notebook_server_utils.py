"""Utilities for managing scribe servers."""

import base64
import re
import subprocess
import sys
import time
from typing import Any

import requests
from fastmcp.utilities.types import Image

from ._image_processing_utils import resize_image_if_needed
from ._image_result_utils import save_image_to_temp


def find_safe_port(start_port=20000, max_port=30000):
    """Find a port that's not in use by anyone.

    Uses random selection to minimize conflicts between users.

    Args:
        start_port: Minimum port number (default: 20000)
        max_port: Maximum port number (default: 30000)

    Returns:
        int: Available port number, or None if none found
    """
    import random
    import socket

    # Try random ports first (more efficient and less likely to conflict)
    ports_to_try = list(range(start_port, max_port + 1))
    random.shuffle(ports_to_try)

    # Try up to 100 random ports
    for port in ports_to_try[:100]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            # Port in use
            continue

    return None


def clean_notebook_for_save(nb):
    """Remove cell IDs and other properties that cause validation warnings."""
    for cell in nb.cells:
        try:
            if hasattr(cell, "id"):
                delattr(cell, "id")
        except AttributeError:
            # Some notebook node types don't support attribute deletion
            pass
    return nb


def check_server_health(port: int) -> dict[str, Any] | None:
    """Check if scribe server is running on given port."""
    try:
        url = f"http://127.0.0.1:{port}/api/scribe/health"
        response = requests.get(url, timeout=1)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def start_scribe_server(
    port: int, token: str, notebook_output_dir: str | None = None
) -> subprocess.Popen:
    """Start a Scribe Jupyter server subprocess.

    Args:
        port: Port number to run server on
        token: Authentication token for the server
        notebook_output_dir: Optional directory for storing notebooks

    Returns:
        subprocess.Popen: The running server process

    Raises:
        Exception: If server fails to start or become ready
    """
    # Start the server process using module import
    cmd = [
        sys.executable,
        "-m",
        "scribe.notebook.notebook_server",
        f"--port={port}",
        "--no-browser",
        "--allow-root",
        f"--ServerApp.token={token}",  # Use provided token for auth
        "--ServerApp.password=",  # No password (empty value)
        "--ServerApp.disable_check_xsrf=True",  # Disable CSRF for API calls
    ]

    # Add notebook output directory if specified
    if notebook_output_dir:
        cmd.extend(["--ScribeServerApp.notebooks_dir", notebook_output_dir])

    import tempfile
    stderr_file = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False, prefix="scribe-server-")
    stderr_path = stderr_file.name
    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=stderr_file
    )

    url = f"http://127.0.0.1:{port}"

    # Wait for server to be ready
    max_attempts = 30  # 30 seconds
    for _ in range(max_attempts):
        # Check if process crashed
        if process.poll() is not None:
            stderr_file.close()
            with open(stderr_path) as f:
                stderr_content = f.read()
            raise Exception(f"Jupyter server crashed. Log: {stderr_path}\n{stderr_content[:500]}")

        try:
            response = requests.get(f"{url}/api/scribe/health", timeout=1)
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    else:
        process.terminate()
        process.wait()
        stderr_file.close()
        with open(stderr_path) as f:
            stderr_content = f.read()
        raise Exception(f"Jupyter server failed to start on port {port}. Log: {stderr_path}\n{stderr_content[:500]}")

    print(f"Jupyter server log: {stderr_path}", file=sys.stderr)
    return process


def cleanup_scribe_server(process: subprocess.Popen) -> None:
    """Clean up a Scribe Jupyter server process.

    Args:
        process: The server process to clean up
    """
    if process:
        print("Shutting down managed Jupyter server...", file=sys.stderr)
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def process_jupyter_outputs(
    outputs: list[dict[str, Any]],
    session_id: str | None = None,
    save_images_locally: bool = False,
    provider: str = None,
) -> tuple[list[dict[str, Any]], list[Image]]:
    """Process Jupyter notebook outputs into MCP format.

    Args:
        outputs: Raw Jupyter notebook output data
        session_id: Session ID for saving images (required if save_images_locally=True)
        save_images_locally: If True, save images to temp directory and return file paths

    Returns:
        Tuple of (processed_outputs, images)
        - If save_images_locally=False: images contains fastmcp.Image objects
        - If save_images_locally=True: processed_outputs contains image file paths, images is empty
    """
    processed_outputs = []
    images = []
    image_count = 0

    for output in outputs:
        if output["output_type"] == "stream":
            processed_outputs.append(
                {"type": "text", "content": output["text"].strip()}
            )
        elif output["output_type"] == "execute_result":
            # Handle different MIME types
            if "image/png" in output.get("data", {}):
                img_data = base64.b64decode(output["data"]["image/png"])

                if save_images_locally:
                    if not session_id:
                        raise ValueError(
                            "session_id is required when save_images_locally=True"
                        )
                    # Save image to temp directory
                    image_path = save_image_to_temp(session_id, img_data, image_count)
                    image_count += 1
                    processed_outputs.append(
                        {
                            "type": "text",
                            "content": f"**AGENT MESSAGE**: Code execution produced an image at: {image_path}. You **MUST** use your tools (e.g. the read_file tool) to view the image before proceeding.",
                        }
                    )
                else:
                    # Resize image if needed to prevent 413 errors
                    resized_img_data = resize_image_if_needed(img_data)
                    # Convert resized PNG data to Image object
                    image = Image(data=resized_img_data)
                    images.append(image)
            elif "text/plain" in output["data"]:
                processed_outputs.append(
                    {"type": "result", "content": output["data"]["text/plain"]}
                )
        elif output["output_type"] == "display_data":
            # Handle display data (like images from .show())
            if "image/png" in output.get("data", {}):
                img_data = base64.b64decode(output["data"]["image/png"])

                if save_images_locally:
                    if not session_id:
                        raise ValueError(
                            "session_id is required when save_images_locally=True"
                        )
                    # Save image to temp directory
                    image_path = save_image_to_temp(session_id, img_data, image_count)
                    image_count += 1
                    processed_outputs.append(
                        {
                            "type": "text",
                            "content": f"**AGENT MESSAGE**: Code execution produced an image at: {image_path}. You **MUST** use your tools (e.g. the read_file tool) to view the image before proceeding.",
                        }
                    )
                else:
                    # Resize image if needed to prevent 413 errors
                    resized_img_data = resize_image_if_needed(img_data)
                    # Convert resized PNG data to Image object
                    image = Image(data=resized_img_data)
                    images.append(image)
            elif "text/plain" in output.get("data", {}):
                processed_outputs.append(
                    {"type": "display", "content": output["data"]["text/plain"]}
                )
        elif output["output_type"] == "error":
            # Clean up traceback by removing ANSI escape codes
            cleaned_traceback = []
            for line in output["traceback"]:
                # Remove ANSI escape sequences
                clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                cleaned_traceback.append(clean_line)

            processed_outputs.append(
                {
                    "type": "error",
                    "name": output["ename"],
                    "message": output["evalue"],
                    "traceback": cleaned_traceback,
                }
            )

    return processed_outputs, images

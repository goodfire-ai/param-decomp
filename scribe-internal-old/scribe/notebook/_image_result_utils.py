"""
Utilities for handling image results from Jupyter notebook execution.

This module provides functionality to save images locally to temporary directories
and clean them up when sessions end or the server shuts down.
"""

import shutil
import time
from pathlib import Path
from typing import List, Optional


def save_image_to_temp(session_id: str, image_data: bytes, image_index: int) -> str:
    """Save image data to temporary directory and return the path.

    Args:
        session_id: The session ID
        image_data: The image data as bytes
        image_index: Index to make filename unique

    Returns:
        The absolute path to the saved image
    """
    # Create temp directory structure
    temp_dir = Path.cwd() / ".scribe" / "tmp" / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save image with timestamp and index
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = temp_dir / f"image_{timestamp}_{image_index}.png"

    with open(image_path, "wb") as f:
        f.write(image_data)

    return str(image_path.absolute())


def cleanup_session_images(session_id: str) -> None:
    """Clean up all images for a specific session.

    Args:
        session_id: The session ID whose images should be cleaned up
    """
    session_dir = Path.cwd() / ".scribe" / "tmp" / session_id
    if session_dir.exists():
        try:
            shutil.rmtree(session_dir)
        except Exception as e:
            # Log error but don't raise - cleanup is best-effort
            print(f"Warning: Failed to cleanup images for session {session_id}: {e}")


def cleanup_all_session_images(session_ids: Optional[List[str]] = None) -> None:
    """Clean up images for multiple sessions or all sessions.

    Args:
        session_ids: List of session IDs to clean up. If None, cleans up all sessions.
    """
    tmp_dir = Path.cwd() / ".scribe" / "tmp"
    if not tmp_dir.exists():
        return

    if session_ids is None:
        # Clean up entire tmp directory
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup all session images: {e}")
    else:
        # Clean up specific sessions
        for session_id in session_ids:
            cleanup_session_images(session_id)


def get_temp_dir_size(session_id: Optional[str] = None) -> int:
    """Get the total size of temporary image files.

    Args:
        session_id: If provided, get size for specific session. Otherwise, get total size.

    Returns:
        Total size in bytes
    """
    if session_id:
        session_dir = Path.cwd() / ".scribe" / "tmp" / session_id
        if not session_dir.exists():
            return 0
        dirs_to_check = [session_dir]
    else:
        tmp_dir = Path.cwd() / ".scribe" / "tmp"
        if not tmp_dir.exists():
            return 0
        dirs_to_check = [tmp_dir]

    total_size = 0
    for dir_path in dirs_to_check:
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except Exception:
                    # Skip files we can't read
                    continue

    return total_size

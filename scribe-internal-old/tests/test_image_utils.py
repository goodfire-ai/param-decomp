"""Tests for image utility functions in _image_result_utils.py"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from scribe.notebook._image_result_utils import (
    save_image_to_temp,
    cleanup_session_images,
    cleanup_all_session_images,
    get_temp_dir_size,
)


class TestSaveImageToTemp:
    """Tests for save_image_to_temp function."""

    def test_saves_image_file(self, tmp_path):
        """Image should be saved to temp directory."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            image_data = b"fake_png_data"
            path = save_image_to_temp("session-123", image_data, 0)

            assert os.path.exists(path)
            with open(path, "rb") as f:
                assert f.read() == image_data

    def test_creates_session_directory(self, tmp_path):
        """Should create session-specific directory."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            save_image_to_temp("my-session", b"data", 0)

            session_dir = tmp_path / ".scribe" / "tmp" / "my-session"
            assert session_dir.exists()
            assert session_dir.is_dir()

    def test_unique_filenames_with_index(self, tmp_path):
        """Different indices should create different files."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            path1 = save_image_to_temp("session", b"data1", 0)
            path2 = save_image_to_temp("session", b"data2", 1)

            assert path1 != path2
            assert os.path.exists(path1)
            assert os.path.exists(path2)

    def test_returns_absolute_path(self, tmp_path):
        """Returned path should be absolute."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            path = save_image_to_temp("session", b"data", 0)
            assert os.path.isabs(path)

    def test_filename_contains_timestamp(self, tmp_path):
        """Filename should contain timestamp."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            path = save_image_to_temp("session", b"data", 0)
            filename = os.path.basename(path)

            # Should match pattern: image_YYYYMMDD_HHMMSS_N.png
            assert filename.startswith("image_")
            assert filename.endswith(".png")
            assert "_0.png" in filename


class TestCleanupSessionImages:
    """Tests for cleanup_session_images function."""

    def test_removes_session_directory(self, tmp_path):
        """Should remove the entire session directory."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            # Create session with images
            save_image_to_temp("session-to-delete", b"data1", 0)
            save_image_to_temp("session-to-delete", b"data2", 1)

            session_dir = tmp_path / ".scribe" / "tmp" / "session-to-delete"
            assert session_dir.exists()

            cleanup_session_images("session-to-delete")
            assert not session_dir.exists()

    def test_handles_nonexistent_session(self, tmp_path):
        """Should not raise when session doesn't exist."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            # Should not raise
            cleanup_session_images("nonexistent-session")

    def test_preserves_other_sessions(self, tmp_path):
        """Should only remove the specified session."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            save_image_to_temp("session-a", b"data", 0)
            save_image_to_temp("session-b", b"data", 0)

            cleanup_session_images("session-a")

            session_a_dir = tmp_path / ".scribe" / "tmp" / "session-a"
            session_b_dir = tmp_path / ".scribe" / "tmp" / "session-b"

            assert not session_a_dir.exists()
            assert session_b_dir.exists()


class TestCleanupAllSessionImages:
    """Tests for cleanup_all_session_images function."""

    def test_cleanup_specific_sessions(self, tmp_path):
        """Should clean up only specified sessions."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            save_image_to_temp("session-1", b"data", 0)
            save_image_to_temp("session-2", b"data", 0)
            save_image_to_temp("session-3", b"data", 0)

            cleanup_all_session_images(["session-1", "session-2"])

            assert not (tmp_path / ".scribe" / "tmp" / "session-1").exists()
            assert not (tmp_path / ".scribe" / "tmp" / "session-2").exists()
            assert (tmp_path / ".scribe" / "tmp" / "session-3").exists()

    def test_cleanup_all_sessions(self, tmp_path):
        """With None, should remove entire tmp directory."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            save_image_to_temp("session-1", b"data", 0)
            save_image_to_temp("session-2", b"data", 0)

            cleanup_all_session_images(None)

            tmp_dir = tmp_path / ".scribe" / "tmp"
            assert not tmp_dir.exists()

    def test_handles_empty_tmp_dir(self, tmp_path):
        """Should not raise when tmp directory doesn't exist."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            # Should not raise
            cleanup_all_session_images(None)
            cleanup_all_session_images(["some-session"])


class TestGetTempDirSize:
    """Tests for get_temp_dir_size function."""

    def test_returns_zero_for_nonexistent(self, tmp_path):
        """Should return 0 when directory doesn't exist."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            size = get_temp_dir_size()
            assert size == 0

    def test_returns_zero_for_nonexistent_session(self, tmp_path):
        """Should return 0 for nonexistent session."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            size = get_temp_dir_size("nonexistent")
            assert size == 0

    def test_returns_file_sizes_for_session(self, tmp_path):
        """Should return total size for session."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            data = b"x" * 100  # 100 bytes
            save_image_to_temp("test-session", data, 0)
            save_image_to_temp("test-session", data, 1)

            size = get_temp_dir_size("test-session")
            assert size == 200

    def test_returns_total_size_all_sessions(self, tmp_path):
        """Should return total size across all sessions."""
        with patch.object(Path, "cwd", return_value=tmp_path):
            data = b"x" * 50  # 50 bytes each
            save_image_to_temp("session-1", data, 0)
            save_image_to_temp("session-2", data, 0)

            size = get_temp_dir_size()
            assert size == 100

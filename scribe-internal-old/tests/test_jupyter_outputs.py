"""Tests for Jupyter output processing in _notebook_server_utils.py"""

import base64
import pytest
from unittest.mock import patch, MagicMock

from scribe.notebook._notebook_server_utils import process_jupyter_outputs


class TestProcessJupyterOutputs:
    """Tests for process_jupyter_outputs function."""

    def test_empty_outputs(self):
        """Empty output list should return empty results."""
        outputs, images = process_jupyter_outputs([])
        assert outputs == []
        assert images == []

    def test_stream_output(self):
        """Stream output (print statements) should be converted to text type."""
        jupyter_outputs = [
            {"output_type": "stream", "text": "Hello, world!\n"}
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert len(outputs) == 1
        assert outputs[0] == {"type": "text", "content": "Hello, world!"}
        assert images == []

    def test_stream_output_strips_whitespace(self):
        """Stream output should have whitespace stripped."""
        jupyter_outputs = [
            {"output_type": "stream", "text": "  some text  \n\n"}
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert outputs[0]["content"] == "some text"

    def test_execute_result_text(self):
        """Execute result with text/plain should be converted to result type."""
        jupyter_outputs = [
            {
                "output_type": "execute_result",
                "data": {"text/plain": "42"},
            }
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert len(outputs) == 1
        assert outputs[0] == {"type": "result", "content": "42"}
        assert images == []

    def test_display_data_text(self):
        """Display data with text/plain should be converted to display type."""
        jupyter_outputs = [
            {
                "output_type": "display_data",
                "data": {"text/plain": "<Figure size 640x480>"},
            }
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert len(outputs) == 1
        assert outputs[0] == {"type": "display", "content": "<Figure size 640x480>"}

    def test_error_output(self):
        """Error output should be converted with cleaned traceback."""
        jupyter_outputs = [
            {
                "output_type": "error",
                "ename": "ValueError",
                "evalue": "invalid value",
                "traceback": [
                    "\x1b[0;31mValueError\x1b[0m: invalid value",
                    "\x1b[0;32m  line 1\x1b[0m",
                ],
            }
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert len(outputs) == 1
        assert outputs[0]["type"] == "error"
        assert outputs[0]["name"] == "ValueError"
        assert outputs[0]["message"] == "invalid value"
        # ANSI codes should be removed
        assert "\x1b" not in outputs[0]["traceback"][0]
        assert "ValueError" in outputs[0]["traceback"][0]

    def test_error_ansi_cleanup(self):
        """Error traceback should have all ANSI codes removed."""
        jupyter_outputs = [
            {
                "output_type": "error",
                "ename": "NameError",
                "evalue": "name 'x' is not defined",
                "traceback": [
                    "\x1b[1;31m---------------------------------------------------------------------------\x1b[0m",
                    "\x1b[1;31mNameError\x1b[0m                                 Traceback (most recent call last)",
                    "\x1b[0;32m<ipython-input-1>\x1b[0m in \x1b[0;36m<module>\x1b[0;34m\x1b[0m",
                    "\x1b[1;32m      1\x1b[0m \x1b[0mprint\x1b[0m\x1b[0;34m(\x1b[0m\x1b[0mx\x1b[0m\x1b[0;34m)\x1b[0m",
                ],
            }
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        for line in outputs[0]["traceback"]:
            assert "\x1b[" not in line
            assert "\x1b" not in line

    def test_multiple_stream_outputs(self):
        """Multiple stream outputs should all be processed."""
        jupyter_outputs = [
            {"output_type": "stream", "text": "Line 1\n"},
            {"output_type": "stream", "text": "Line 2\n"},
            {"output_type": "stream", "text": "Line 3\n"},
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert len(outputs) == 3
        assert all(o["type"] == "text" for o in outputs)
        assert outputs[0]["content"] == "Line 1"
        assert outputs[1]["content"] == "Line 2"
        assert outputs[2]["content"] == "Line 3"

    def test_mixed_output_types(self):
        """Mixed output types should be processed in order."""
        jupyter_outputs = [
            {"output_type": "stream", "text": "Starting...\n"},
            {"output_type": "stream", "text": "Done!\n"},
            {"output_type": "execute_result", "data": {"text/plain": "100"}},
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert len(outputs) == 3
        assert outputs[0] == {"type": "text", "content": "Starting..."}
        assert outputs[1] == {"type": "text", "content": "Done!"}
        assert outputs[2] == {"type": "result", "content": "100"}

    @patch("scribe.notebook._notebook_server_utils.resize_image_if_needed")
    def test_execute_result_image(self, mock_resize):
        """Execute result with image/png should create Image object."""
        # Create a simple PNG-like data (1x1 red pixel)
        png_data = base64.b64encode(b"fake_png_data").decode()
        mock_resize.return_value = b"fake_png_data"

        jupyter_outputs = [
            {
                "output_type": "execute_result",
                "data": {"image/png": png_data},
            }
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert len(outputs) == 0
        assert len(images) == 1
        mock_resize.assert_called_once()

    @patch("scribe.notebook._notebook_server_utils.resize_image_if_needed")
    def test_display_data_image(self, mock_resize):
        """Display data with image/png should create Image object."""
        png_data = base64.b64encode(b"fake_png_data").decode()
        mock_resize.return_value = b"fake_png_data"

        jupyter_outputs = [
            {
                "output_type": "display_data",
                "data": {"image/png": png_data},
            }
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert len(outputs) == 0
        assert len(images) == 1

    @patch("scribe.notebook._notebook_server_utils.save_image_to_temp")
    def test_image_save_locally_execute_result(self, mock_save):
        """With save_images_locally=True, images should be saved to temp."""
        mock_save.return_value = "/tmp/test/image_0.png"
        png_data = base64.b64encode(b"fake_png_data").decode()

        jupyter_outputs = [
            {
                "output_type": "execute_result",
                "data": {"image/png": png_data},
            }
        ]
        outputs, images = process_jupyter_outputs(
            jupyter_outputs,
            session_id="test-session",
            save_images_locally=True,
        )
        assert len(outputs) == 1
        assert len(images) == 0
        assert "image at: /tmp/test/image_0.png" in outputs[0]["content"]
        mock_save.assert_called_once()

    @patch("scribe.notebook._notebook_server_utils.save_image_to_temp")
    def test_image_save_locally_display_data(self, mock_save):
        """Display data images should also save locally when enabled."""
        mock_save.return_value = "/tmp/test/image_0.png"
        png_data = base64.b64encode(b"fake_png_data").decode()

        jupyter_outputs = [
            {
                "output_type": "display_data",
                "data": {"image/png": png_data},
            }
        ]
        outputs, images = process_jupyter_outputs(
            jupyter_outputs,
            session_id="test-session",
            save_images_locally=True,
        )
        assert len(outputs) == 1
        assert len(images) == 0
        assert "/tmp/test/image_0.png" in outputs[0]["content"]

    def test_save_images_locally_requires_session_id(self):
        """save_images_locally=True without session_id should raise."""
        png_data = base64.b64encode(b"fake_png_data").decode()

        jupyter_outputs = [
            {
                "output_type": "execute_result",
                "data": {"image/png": png_data},
            }
        ]
        with pytest.raises(ValueError, match="session_id is required"):
            process_jupyter_outputs(
                jupyter_outputs,
                session_id=None,
                save_images_locally=True,
            )

    @patch("scribe.notebook._notebook_server_utils.save_image_to_temp")
    def test_multiple_images_increments_index(self, mock_save):
        """Multiple images should have incrementing indices."""
        mock_save.side_effect = [
            "/tmp/test/image_0.png",
            "/tmp/test/image_1.png",
        ]
        png_data = base64.b64encode(b"fake_png_data").decode()

        jupyter_outputs = [
            {"output_type": "display_data", "data": {"image/png": png_data}},
            {"output_type": "display_data", "data": {"image/png": png_data}},
        ]
        outputs, images = process_jupyter_outputs(
            jupyter_outputs,
            session_id="test-session",
            save_images_locally=True,
        )
        assert len(outputs) == 2
        # Check that save was called with incrementing indices
        calls = mock_save.call_args_list
        assert calls[0][0][2] == 0  # First call, index 0
        assert calls[1][0][2] == 1  # Second call, index 1

    def test_unknown_output_type_ignored(self):
        """Unknown output types should be silently ignored."""
        jupyter_outputs = [
            {"output_type": "unknown_type", "data": {"foo": "bar"}},
            {"output_type": "stream", "text": "valid\n"},
        ]
        outputs, images = process_jupyter_outputs(jupyter_outputs)
        assert len(outputs) == 1
        assert outputs[0]["content"] == "valid"

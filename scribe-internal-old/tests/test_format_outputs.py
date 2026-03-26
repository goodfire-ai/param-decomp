"""Tests for output formatting and file parsing."""

from unittest.mock import MagicMock

from scribe.notebook._output_formatting import format_outputs_compact
from scribe.notebook._file_parser import parse_named_cells


class TestFormatOutputsCompact:
    """Tests for format_outputs_compact function."""

    def test_empty_outputs_returns_no_output(self):
        """Empty outputs should return '(no output)'."""
        result = format_outputs_compact([], [])
        assert result == ["(no output)"]

    def test_simple_text_output(self):
        """Print statement output should be returned as-is."""
        outputs = [{"type": "text", "content": "Hello, world!"}]
        result = format_outputs_compact(outputs, [])
        assert result == ["Hello, world!"]

    def test_multiple_text_outputs_concatenated(self):
        """Multiple text outputs should be joined with newlines."""
        outputs = [
            {"type": "text", "content": "Line 1"},
            {"type": "text", "content": "Line 2"},
        ]
        result = format_outputs_compact(outputs, [])
        assert result == ["Line 1\nLine 2"]

    def test_result_output_with_arrow(self):
        """Result values should be prefixed with '=>'."""
        outputs = [{"type": "result", "content": "42"}]
        result = format_outputs_compact(outputs, [])
        assert result == ["=> 42"]

    def test_result_none_ignored(self):
        """Result of 'None' should be ignored."""
        outputs = [{"type": "result", "content": "None"}]
        result = format_outputs_compact(outputs, [])
        assert result == ["(no output)"]

    def test_result_empty_ignored(self):
        """Empty result should be ignored."""
        outputs = [{"type": "result", "content": ""}]
        result = format_outputs_compact(outputs, [])
        assert result == ["(no output)"]

    def test_display_output(self):
        """Display data should be included."""
        outputs = [{"type": "display", "content": "Some display data"}]
        result = format_outputs_compact(outputs, [])
        assert result == ["Some display data"]

    def test_text_and_result_combined(self):
        """Text and result outputs should be combined."""
        outputs = [
            {"type": "text", "content": "Processing..."},
            {"type": "result", "content": "[1, 2, 3]"},
        ]
        result = format_outputs_compact(outputs, [])
        assert result == ["Processing...\n=> [1, 2, 3]"]

    def test_error_basic_format(self):
        """Error should be formatted with name and message."""
        outputs = [
            {
                "type": "error",
                "name": "ValueError",
                "message": "invalid literal",
                "traceback": [],
            }
        ]
        result = format_outputs_compact(outputs, [])
        assert len(result) == 1
        assert "ERROR: ValueError: invalid literal" in result[0]

    def test_error_with_traceback_location(self):
        """Error should extract location from traceback."""
        outputs = [
            {
                "type": "error",
                "name": "NameError",
                "message": "name 'x' is not defined",
                "traceback": [
                    "Traceback (most recent call last):",
                    "  Cell In[1], line 5, in my_function",
                    "    print(x)",
                    "NameError: name 'x' is not defined",
                ],
            }
        ]
        result = format_outputs_compact(outputs, [])
        assert len(result) == 1
        assert "ERROR: NameError" in result[0]
        assert "line 5" in result[0]

    def test_images_appended_to_result(self):
        """Images should be appended after text content."""
        outputs = [{"type": "text", "content": "Plot generated"}]
        mock_image = MagicMock()
        images = [mock_image]
        result = format_outputs_compact(outputs, images)
        assert len(result) == 2
        assert result[0] == "Plot generated"
        assert result[1] is mock_image

    def test_images_only_adds_count_prefix(self):
        """When only images exist, add count prefix."""
        mock_image1 = MagicMock()
        mock_image2 = MagicMock()
        images = [mock_image1, mock_image2]
        result = format_outputs_compact([], images)
        assert len(result) == 3
        assert "[2 image(s) returned]" in result[0]
        assert result[1] is mock_image1
        assert result[2] is mock_image2

    def test_single_image_only(self):
        """Single image should also get count prefix."""
        mock_image = MagicMock()
        images = [mock_image]
        result = format_outputs_compact([], images)
        assert len(result) == 2
        assert "[1 image(s) returned]" in result[0]

    def test_empty_content_text_ignored(self):
        """Empty text content should be ignored."""
        outputs = [{"type": "text", "content": ""}]
        result = format_outputs_compact(outputs, [])
        assert result == ["(no output)"]

    def test_large_output_truncated(self):
        """Output exceeding 10K chars should be truncated."""
        big_text = "x" * 20_000
        outputs = [{"type": "text", "content": big_text}]
        result = format_outputs_compact(outputs, [])
        assert len(result) == 1
        assert len(result[0]) < 15_000
        assert "truncated" in result[0]
        assert "10,000 chars" in result[0]
        assert "see notebook" in result[0]

    def test_mixed_outputs_with_images(self):
        """Complex case with text, result, and images."""
        outputs = [
            {"type": "text", "content": "Starting analysis..."},
            {"type": "text", "content": "Done!"},
            {"type": "result", "content": "{'accuracy': 0.95}"},
        ]
        mock_image = MagicMock()
        images = [mock_image]
        result = format_outputs_compact(outputs, images)
        assert len(result) == 2
        assert "Starting analysis..." in result[0]
        assert "Done!" in result[0]
        assert "=> {'accuracy': 0.95}" in result[0]
        assert result[1] is mock_image

    def test_error_with_relevant_code_line(self):
        """Error should try to extract relevant code from traceback."""
        outputs = [
            {
                "type": "error",
                "name": "TypeError",
                "message": "unsupported operand type(s)",
                "traceback": [
                    "Traceback (most recent call last):",
                    "  File \"<stdin>\", line 1, in <module>",
                    "    result = 'hello' + 42",
                    "TypeError: unsupported operand type(s)",
                ],
            }
        ]
        result = format_outputs_compact(outputs, [])
        assert "TypeError" in result[0]
        # Should contain the relevant code line
        assert "unsupported operand" in result[0]

    def test_multiple_errors(self):
        """Multiple errors should all be included."""
        outputs = [
            {
                "type": "error",
                "name": "Error1",
                "message": "first error",
                "traceback": [],
            },
            {
                "type": "error",
                "name": "Error2",
                "message": "second error",
                "traceback": [],
            },
        ]
        result = format_outputs_compact(outputs, [])
        assert len(result) == 1
        assert "Error1" in result[0]
        assert "Error2" in result[0]


class TestParseNamedCells:
    """Tests for parse_named_cells function."""

    def test_single_cell(self):
        content = "# %% imports\nimport numpy as np"
        cells = parse_named_cells(content)
        assert len(cells) == 1
        assert cells[0].tag == "imports"
        assert cells[0].cell_type == "code"
        assert cells[0].source == "import numpy as np"

    def test_two_cells(self):
        content = "# %% imports\nimport numpy as np\n\n# %% analysis\nprint('hi')"
        cells = parse_named_cells(content)
        assert len(cells) == 2
        assert cells[0].tag == "imports"
        assert cells[0].cell_type == "code"
        assert cells[0].source == "import numpy as np"
        assert cells[1].tag == "analysis"
        assert cells[1].cell_type == "code"
        assert cells[1].source == "print('hi')"

    def test_content_before_first_marker_ignored(self):
        content = "# header comment\nx = 1\n\n# %% setup\nimport numpy"
        cells = parse_named_cells(content)
        assert len(cells) == 1
        assert cells[0].tag == "setup"
        assert cells[0].source == "import numpy"

    def test_empty_body_cell(self):
        content = "# %% empty\n\n# %% real\nprint('hi')"
        cells = parse_named_cells(content)
        assert len(cells) == 2
        assert cells[0].tag == "empty"
        assert cells[0].source == ""
        assert cells[1].tag == "real"
        assert cells[1].source == "print('hi')"

    def test_no_markers_returns_empty(self):
        content = "import numpy as np\nprint('hello')"
        cells = parse_named_cells(content)
        assert cells == []

    def test_multiline_cells(self):
        content = "# %% imports\nimport numpy as np\nimport torch\n\nx = np.array([1, 2, 3])\n\n# %% run\nprint(x)\nprint(x.sum())"
        cells = parse_named_cells(content)
        assert len(cells) == 2
        assert "import numpy" in cells[0].source
        assert "import torch" in cells[0].source
        assert "print(x)" in cells[1].source

    def test_tag_required(self):
        import pytest
        content = "# %%\nimport numpy"
        with pytest.raises(AssertionError, match="must have a tag"):
            parse_named_cells(content)

    def test_duplicate_tags_rejected(self):
        import pytest
        content = "# %% setup\nx = 1\n# %% setup\ny = 2"
        with pytest.raises(AssertionError, match="Duplicate"):
            parse_named_cells(content)

    def test_hyphenated_tags(self):
        content = "# %% load-model\nmodel = load()\n# %% run-inference\nout = model(x)"
        cells = parse_named_cells(content)
        assert cells[0].tag == "load-model"
        assert cells[1].tag == "run-inference"

    def test_markdown_cell(self):
        content = "# %% md:intro\n# This is markdown\n# Second line"
        cells = parse_named_cells(content)
        assert len(cells) == 1
        assert cells[0].tag == "intro"
        assert cells[0].cell_type == "markdown"
        assert cells[0].source == "This is markdown\nSecond line"

    def test_mixed_code_and_markdown(self):
        content = "# %% md:header\n# My Title\n\n# %% imports\nimport numpy as np\n\n# %% md:notes\n# Some notes"
        cells = parse_named_cells(content)
        assert len(cells) == 3
        assert cells[0].tag == "header"
        assert cells[0].cell_type == "markdown"
        assert cells[0].source == "My Title"
        assert cells[1].tag == "imports"
        assert cells[1].cell_type == "code"
        assert cells[1].source == "import numpy as np"
        assert cells[2].tag == "notes"
        assert cells[2].cell_type == "markdown"
        assert cells[2].source == "Some notes"

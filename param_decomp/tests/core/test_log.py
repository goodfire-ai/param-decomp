import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import override

import pytest

from param_decomp.core.log import logger, setup_console_logger, setup_logger


@pytest.fixture
def isolated_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(logger, "handlers", [logging.NullHandler()])
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    monkeypatch.setattr(logger, "disabled", False)
    monkeypatch.setattr(logger, "propagate", True)
    yield
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


@pytest.mark.usefixtures("isolated_logger")
def test_setup_console_logger_is_idempotent_and_writes_once_to_stdout(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root_logger = logging.getLogger()
    monkeypatch.setattr(root_logger, "handlers", [logging.StreamHandler()])
    monkeypatch.setattr(root_logger, "level", logging.INFO)
    logger.disabled = True

    setup_console_logger()
    setup_console_logger()
    logger.values({"Run ID": "p-00000000", "Job ID": "123"})

    captured = capsys.readouterr()
    assert captured.out == "\n  Run ID : p-00000000\n  Job ID : 123\n"
    assert captured.err == ""
    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO
    assert not logger.disabled
    assert not logger.propagate


@pytest.mark.usefixtures("isolated_logger")
def test_setup_logger_replaces_handlers_without_changing_its_output_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logfile = tmp_path / "nested" / "run.log"

    setup_logger(logfile)
    setup_logger(logfile)
    logger.info("info message")
    logger.warning("warning message")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - INFO - info message\n"
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - WARNING - warning message\n",
        captured.err,
    )
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - WARNING - warning message\n",
        logfile.read_text(),
    )
    assert len(logger.handlers) == 2
    assert logger.propagate


class _CloseTrackingHandler(logging.NullHandler):
    def __init__(self) -> None:
        super().__init__()
        self.closed_by_setup = False

    @override
    def close(self) -> None:
        self.closed_by_setup = True
        super().close()


@pytest.mark.usefixtures("isolated_logger")
def test_setup_does_not_close_unrelated_logger_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    unrelated_logger = logging.getLogger("unrelated-test-logger")
    unrelated_handler = _CloseTrackingHandler()
    monkeypatch.setattr(unrelated_logger, "handlers", [unrelated_handler])

    setup_console_logger()

    assert unrelated_logger.handlers == [unrelated_handler]
    assert not unrelated_handler.closed_by_setup

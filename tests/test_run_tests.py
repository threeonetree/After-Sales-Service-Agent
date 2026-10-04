from pathlib import Path
from unittest.mock import Mock
import sys

import pytest

from scripts import run_tests


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tests, "ROOT", tmp_path)
    monkeypatch.setattr(run_tests.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(run_tests.shutil, "which", lambda name: "allure")
    opened = Mock()
    monkeypatch.setattr(run_tests.webbrowser, "open", opened)
    return tmp_path, opened


def test_failed_tests_still_generate_report_and_preserve_failure(runner, monkeypatch):
    root, opened = runner
    commands = []
    def execute(command, log):
        commands.append(command)
        log.write_text("AssertionError: expected value\n", encoding="utf-8")
        if command[0] == "allure":
            index = root / "reports/allure-report/index.html"
            index.parent.mkdir()
            index.write_text("report")
            return 0
        return 1
    monkeypatch.setattr(run_tests, "run_logged", execute)
    assert run_tests.main(["--open-report", "--", "tests/test_image_input.py"]) == 1
    assert commands[0][:3] == [sys.executable, "-m", "pytest"]
    assert commands[0][-1] == "tests/test_image_input.py"
    assert "--single-file" in commands[1]
    assert (root / "reports/pytest.log").exists()
    assert "offline" in (root / "reports/allure-results/environment.properties").read_text()
    opened.assert_called_once()


def test_missing_cli_keeps_results_and_does_not_open_stale_report(runner, monkeypatch, capsys):
    root, opened = runner
    old = root / "reports/allure-report"
    old.mkdir(parents=True)
    (old / "index.html").write_text("stale success")
    monkeypatch.setattr(run_tests.shutil, "which", lambda name: None)
    monkeypatch.setattr(run_tests, "run_logged", lambda *args: 0)
    assert run_tests.main(["--open-report"]) == 2
    assert not old.exists()
    assert "docs/testing.md" in capsys.readouterr().out
    opened.assert_not_called()


def test_generation_failure_never_overrides_test_failure(runner, monkeypatch):
    _, opened = runner
    monkeypatch.setattr(run_tests, "run_logged", Mock(side_effect=[1, 3]))
    assert run_tests.main(["--open-report"]) == 1
    opened.assert_not_called()


def test_plain_runner_needs_no_java_or_allure_cli(runner, monkeypatch):
    _, opened = runner
    monkeypatch.setattr(run_tests, "run_logged", lambda *args: 0)
    lookup = Mock()
    monkeypatch.setattr(run_tests.shutil, "which", lookup)
    assert run_tests.main([]) == 0
    lookup.assert_not_called()
    opened.assert_not_called()


def test_logger_captures_stdout_stderr_and_exit_code(tmp_path):
    log = tmp_path / "pytest.log"
    code = run_tests.run_logged([
        sys.executable, "-c", "import sys; print('test output'); print('error detail', file=sys.stderr); sys.exit(1)",
    ], log)
    assert code == 1
    assert "test output" in log.read_text() and "error detail" in log.read_text()

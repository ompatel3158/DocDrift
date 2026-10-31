"""Unit and integration tests for CodeHealthVerifier and docdrift test CLI command."""

from pathlib import Path
from typer.testing import CliRunner

from docdrift.cli import app
from docdrift.verifiers.code_runner import CodeHealthVerifier, CodeHealthReport

runner = CliRunner()


class TestCodeHealthVerifier:
    def test_check_codebase_health_clean_repo(self, tmp_path: Path):
        # Create valid python file and valid json
        (tmp_path / "valid.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
        (tmp_path / "package.json").write_text('{"name": "clean-test", "version": "1.0.0"}', encoding="utf-8")

        report = CodeHealthVerifier.check_codebase_health(str(tmp_path), run_tests=False)
        assert isinstance(report, CodeHealthReport)
        assert report.is_code_working is True
        assert report.syntax_valid_count >= 2
        assert len(report.syntax_errors) == 0

    def test_check_codebase_health_detects_syntax_error(self, tmp_path: Path):
        # Create broken python file
        (tmp_path / "broken.py").write_text("def broken(\n    return 42\n", encoding="utf-8")

        report = CodeHealthVerifier.check_codebase_health(str(tmp_path), run_tests=False)
        assert report.is_code_working is False
        assert len(report.syntax_errors) >= 1
        assert any("broken.py" in err["file"] for err in report.syntax_errors)

    def test_check_codebase_health_entrypoint_detection(self, tmp_path: Path):
        # Create entrypoint file with exported symbols
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "main.py").write_text("import os\napp = 'FastAPI'\nserver = 'ASGI'\n", encoding="utf-8")

        report = CodeHealthVerifier.check_codebase_health(str(tmp_path), run_tests=False)
        assert any(ep["entrypoint"] == "app/main.py" for ep in report.entrypoints_checked)
        ep = next(ep for ep in report.entrypoints_checked if ep["entrypoint"] == "app/main.py")
        assert ep["syntax_valid"] is True
        assert "app" in ep["exported_symbols"]

    def test_check_codebase_health_missing_dependencies(self, tmp_path: Path):
        # Create requirements.txt lacking redis
        (tmp_path / "requirements.txt").write_text("fastapi==0.115.0\n", encoding="utf-8")
        (tmp_path / "worker.py").write_text("import fastapi\nimport redis\n", encoding="utf-8")

        report = CodeHealthVerifier.check_codebase_health(str(tmp_path), run_tests=False)
        assert "redis" in report.missing_dependencies
        assert report.is_code_working is False

    def test_cli_test_command(self, tmp_path: Path):
        (tmp_path / "main.py").write_text("print('hello world')\n", encoding="utf-8")
        res = runner.invoke(app, ["test", str(tmp_path), "--no-run-tests"])
        assert res.exit_code == 0
        assert "HEALTHY & EXECUTABLE" in res.output or "Syntax & Compilation" in res.output

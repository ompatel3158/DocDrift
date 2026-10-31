"""End-to-end CLI integration tests for DocDrift."""

import json
import os
import shutil
import tempfile
from pathlib import Path
import pytest
from typer.testing import CliRunner

from docdrift.cli import app

runner = CliRunner()
FIXTURES_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "fixtures", "sample_drifted_repo"))


def test_cli_help():
    """Verify CLI help text and available subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "DocDrift" in result.output or "docdrift" in result.output
    assert "check" in result.output
    assert "review" in result.output
    assert "watch" in result.output
    assert "memory" in result.output


def test_cli_check_detects_drift_offline():
    """Verify docdrift check detects documentation drift in sample repo in offline mode."""
    result = runner.invoke(app, ["check", FIXTURES_REPO, "--offline"])
    # Should exit with code 1 when drift is detected without --fix
    assert result.exit_code == 1
    output = result.output
    assert "DRIFTED" in output or "drifted" in output
    assert "claims" in output.lower() or "drift" in output.lower()


def test_cli_check_with_diff_and_pr():
    """Verify --diff and --pr flags generate unified diff and PR description."""
    result = runner.invoke(app, ["check", FIXTURES_REPO, "--offline", "--diff", "--pr"])
    assert result.exit_code == 1
    output = result.output
    assert "Git Unified Diff" in output or "---" in output or "+++" in output
    assert "GitHub PR Description" in output or "DocDrift Automated Documentation Fix" in output


def test_cli_check_json_output(tmp_path: Path):
    """Verify --json exports structured report JSON."""
    json_file = str(tmp_path / "drift_report.json")
    result = runner.invoke(app, ["check", FIXTURES_REPO, "--offline", "--json", json_file])
    assert result.exit_code == 1
    assert os.path.exists(json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "total_claims" in data
    assert "drifted_count" in data
    assert "claims" in data
    assert data["drifted_count"] > 0
    assert data["total_claims"] >= data["drifted_count"]


def test_cli_memory_inspect_and_clear():
    """Verify docdrift memory command displays statistics and supports --clear."""
    # First inspect stats
    res_stats = runner.invoke(app, ["memory", FIXTURES_REPO])
    assert res_stats.exit_code == 0
    assert "DocDrift Memory Engine Stats" in res_stats.output or "Backend" in res_stats.output

    # Then clear memory
    res_clear = runner.invoke(app, ["memory", FIXTURES_REPO, "--clear"])
    assert res_clear.exit_code == 0
    assert "Cleared DocDrift memory cache" in res_clear.output


def test_cli_check_with_fix_applies_modifications(tmp_path: Path):
    """Verify --fix modifies drifted documentation files in place."""
    # Copy fixture repo to temporary directory so tests don't permanently alter fixtures
    temp_repo = tmp_path / "test_repo"
    shutil.copytree(FIXTURES_REPO, temp_repo)

    readme_path = temp_repo / "README.md"
    original_readme = readme_path.read_text(encoding="utf-8")
    assert "npm run dev" in original_readme

    # Run check with --fix
    result = runner.invoke(app, ["check", str(temp_repo), "--offline", "--fix"])
    # When --fix is specified, exit code is 0 on successful application
    assert result.exit_code == 0
    assert "Successfully updated" in result.output

    # Verify that README.md has been modified
    updated_readme = readme_path.read_text(encoding="utf-8")
    assert updated_readme != original_readme
    assert "npm run start" in updated_readme or "npm start" in updated_readme or "start" in updated_readme


def test_cli_review_no_drift_when_clean(tmp_path: Path):
    """Verify review on empty/clean repo handles zero drift gracefully."""
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    readme = empty_repo / "README.md"
    readme.write_text("# Clean Project\n\nWelcome to our clean documentation.\n", encoding="utf-8")

    result = runner.invoke(app, ["review", str(empty_repo), "--offline"])
    assert result.exit_code == 0
    assert "No documentation drift found" in result.output

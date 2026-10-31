"""Tests for DocDrift code-level fix constraints and validation."""

import os
from pathlib import Path
import pytest

from docdrift.models import Claim, ClaimType, Verdict
from docdrift.fixer import validate_fix, RuleBasedFixer, GemmaFixer


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Creates a mock repo with package.json and source files."""
    repo = tmp_path / "mock_repo"
    repo.mkdir()
    
    # package.json with scripts
    pkg = repo / "package.json"
    pkg.write_text(
        '{\n  "name": "mock-app",\n  "scripts": {\n    "start": "node index.js",\n    "test": "jest",\n    "build": "tsc"\n  }\n}',
        encoding="utf-8"
    )

    # Actual files
    src = repo / "src"
    src.mkdir()
    (src / "app.js").write_text("// app code", encoding="utf-8")
    (repo / "config.json").write_text("{}", encoding="utf-8")

    return repo


def test_reject_command_replaced_by_directory_or_path(sample_repo: Path):
    """Reject replacing a command script with a directory path like 'examples/cookies'."""
    claim = Claim(
        doc_path=str(sample_repo / "README.md"),
        line_number=10,
        raw_line="$ express /tmp/foo && cd /tmp/foo",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="express /tmp/foo && cd /tmp/foo",
        verdict=Verdict.DRIFTED,
        evidence="Path '/tmp/foo' not found.",
    )
    
    # Attempting to replace command with a directory
    is_valid, reason = validate_fix(claim, "examples/cookies", repo_path=str(sample_repo))
    assert not is_valid
    assert "file or directory path, not a command" in reason


def test_reject_server_startup_replaced_by_test_command(sample_repo: Path):
    """Reject replacing a server startup/dev command with a test command."""
    claim = Claim(
        doc_path=str(sample_repo / "README.md"),
        line_number=15,
        raw_line="Run dev server: `npm run dev`",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run dev",
        verdict=Verdict.DRIFTED,
        evidence="Script 'dev' not found in package.json.",
    )

    # Attempting to replace dev with test
    is_valid, reason = validate_fix(claim, "Run dev server: `npm test`", repo_path=str(sample_repo))
    assert not is_valid
    assert "Cannot replace server startup command with a test command" in reason


def test_reject_invalid_npm_syntax(sample_repo: Path):
    """Reject custom scripts invoked without 'run' (e.g. 'npm test-tap')."""
    claim = Claim(
        doc_path=str(sample_repo / "README.md"),
        line_number=20,
        raw_line="`npm start`",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm start",
        verdict=Verdict.DRIFTED,
        evidence="Script not found.",
    )

    is_valid, reason = validate_fix(claim, "`npm test-tap`", repo_path=str(sample_repo))
    assert not is_valid
    assert "Invalid npm syntax" in reason


def test_reject_nonexistent_file_path(sample_repo: Path):
    """Reject replacing a file path with another path that does not exist in repo."""
    claim = Claim(
        doc_path=str(sample_repo / "README.md"),
        line_number=25,
        raw_line="Settings in `config/old_settings.json`",
        claim_type=ClaimType.FILE_PATH,
        extracted_value="config/old_settings.json",
        verdict=Verdict.DRIFTED,
        evidence="Path not found.",
    )

    # config/nonexistent.json does not exist
    is_valid, reason = validate_fix(claim, "Settings in `config/nonexistent.json`", repo_path=str(sample_repo))
    assert not is_valid
    assert "does not exist in repository" in reason


def test_accept_valid_file_and_command_fixes(sample_repo: Path):
    """Accept verified valid fixes that match code constraints."""
    # 1. Valid command: replace dev with start (both server commands, start is in package.json)
    cmd_claim = Claim(
        doc_path=str(sample_repo / "README.md"),
        line_number=5,
        raw_line="Start with `npm run dev`",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run dev",
        verdict=Verdict.DRIFTED,
        evidence="Script 'dev' not found.",
    )
    is_valid, reason = validate_fix(cmd_claim, "Start with `npm run start`", repo_path=str(sample_repo))
    assert is_valid

    # 2. Valid file path: config.json exists in sample_repo
    file_claim = Claim(
        doc_path=str(sample_repo / "README.md"),
        line_number=8,
        raw_line="Check `settings.json`",
        claim_type=ClaimType.FILE_PATH,
        extracted_value="settings.json",
        verdict=Verdict.DRIFTED,
        evidence="Path not found.",
    )
    is_valid, reason = validate_fix(file_claim, "Check `config.json`", repo_path=str(sample_repo))
    assert is_valid


def test_rule_based_fixer_sets_needs_review_when_invalid(sample_repo: Path):
    """RuleBasedFixer sets needs_review=True when proposed fix fails validation."""
    claim = Claim(
        doc_path=str(sample_repo / "README.md"),
        line_number=10,
        raw_line="Start: `npm run dev`",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run dev",
        verdict=Verdict.DRIFTED,
        evidence="Did you mean `test`?",  # Suggests test for dev!
        suggested_replacement="npm test",
    )

    fixer = RuleBasedFixer(repo_path=str(sample_repo))
    result = fixer.fix_claim(claim, repo_path=str(sample_repo))

    assert result is None
    assert claim.fixed_line is None
    assert claim.needs_review is True
    assert "Cannot replace server startup command" in claim.metadata["rejection_reason"]

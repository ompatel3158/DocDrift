"""Unit and integration tests for InteractiveChat CLI session."""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from docdrift.chat import InteractiveChat
from docdrift.models import Claim, ClaimType, DriftReport, Verdict


@pytest.fixture
def mock_console():
    return Console(file=io.StringIO(), color_system=None, force_terminal=False, width=120)


@pytest.fixture
def sample_repo(tmp_path: Path):
    # Setup a sample repo with package.json and README.md
    pkg = tmp_path / "package.json"
    pkg.write_text('{"scripts": {"start": "node server.js", "test": "jest"}}', encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Test Repo\n\nRun server: `npm run dev`\n", encoding="utf-8")
    env_ex = tmp_path / ".env.example"
    env_ex.write_text("PORT=3000\nAPI_KEY=xyz\n", encoding="utf-8")
    server = tmp_path / "server.js"
    server.write_text("console.log('running');", encoding="utf-8")
    return tmp_path


def test_interactive_chat_initialization(sample_repo, mock_console):
    chat = InteractiveChat(
        repo_path=str(sample_repo),
        offline=True,
        console=mock_console,
    )
    assert os.path.samefile(chat.repo_path, sample_repo)
    assert len(chat.doc_files) >= 1
    assert chat.offline is True


def test_interactive_chat_nonexistent_repo(mock_console):
    with pytest.raises(FileNotFoundError):
        InteractiveChat(repo_path="/non/existent/path/xyz_123", offline=True, console=mock_console)


def test_cmd_help(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    chat._cmd_help()
    output = mock_console.file.getvalue()
    assert "/check" in output
    assert "/drift" in output
    assert "/fix" in output
    assert "/verify" in output


def test_cmd_files(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    chat._cmd_files()
    output = mock_console.file.getvalue()
    assert "README.md" in output
    assert "package.json" in output


def test_cmd_manifests(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    chat._cmd_manifests()
    output = mock_console.file.getvalue()
    assert "start" in output
    assert "test" in output


def test_cmd_verify_valid_command(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    chat._cmd_verify("npm test")
    output = mock_console.file.getvalue()
    assert "VERIFIED" in output


def test_cmd_verify_drifted_command(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    chat._cmd_verify("npm run non_existent_script")
    output = mock_console.file.getvalue()
    assert "DRIFTED" in output


def test_cmd_verify_empty(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    chat._cmd_verify("")
    output = mock_console.file.getvalue()
    assert "Usage: /verify" in output


def test_cmd_drift_when_no_report(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    with patch.object(chat, "_cmd_check") as mock_check:
        chat._cmd_drift()
        mock_check.assert_called_once()


def test_cmd_drift_display(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    drifted_claim = Claim(
        doc_path="README.md",
        line_number=3,
        raw_line="npm run dev",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run dev",
        verdict=Verdict.DRIFTED,
        evidence="Script 'dev' not found in package.json",
        fixed_line="npm run start",
    )
    chat.current_report = DriftReport(
        repo_path=str(sample_repo),
        files_scanned=["README.md"],
        total_claims=1,
        verified_count=0,
        drifted_count=1,
        unknown_count=0,
        claims=[drifted_claim],
    )
    chat._cmd_drift()
    output = mock_console.file.getvalue()
    assert "npm run dev" in output
    assert "npm run start" in output


def test_cmd_diff_and_pr_no_drift(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    chat.current_report = DriftReport(
        repo_path=str(sample_repo),
        files_scanned=["README.md"],
        total_claims=1,
        verified_count=1,
        drifted_count=0,
        unknown_count=0,
        claims=[],
    )
    chat._cmd_diff()
    chat._cmd_pr()
    output = mock_console.file.getvalue()
    assert "No diff to display" in output
    assert "All documentation matches codebase perfectly" in output


def test_natural_language_queries(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)

    # Question about scripts
    ans1 = chat._offline_answer("What npm scripts are available?")
    assert "npm run start" in ans1 or "start" in ans1

    # Question about environment variables
    ans2 = chat._offline_answer("What env vars are configured?")
    assert "PORT" in ans2 or "API_KEY" in ans2

    # Question about docs
    ans3 = chat._offline_answer("What readme files exist?")
    assert "README.md" in ans3

    # Question about project overview
    ans4 = chat._offline_answer("What is this project about?")
    assert "Project Overview" in ans4 or "Sample App" in ans4


def test_chat_slash_test_command(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    chat._handle_slash_command("/test")
    output = mock_console.file.getvalue()
    assert "Codebase Health & Execution Integrity" in output or "Integrity" in output


def test_chat_repo_name_from_git_url(mock_console):
    with patch("docdrift.git_helper.GitHelper.clone_repo", return_value=(os.getcwd(), True)):
        chat = InteractiveChat(repo_path="https://github.com/anmolchaudhary617-ux/CivicPulse", offline=True, console=mock_console)
        assert chat.repo_name == "CivicPulse"


def test_chat_loop_exit_command(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    with patch.object(chat.console, "input", side_effect=["/help", "/exit"]):
        chat.start()
    output = mock_console.file.getvalue()
    assert "DocDrift Interactive CLI Chat" in output
    assert "Exiting DocDrift chat session" in output


def test_chat_loop_keyboard_interrupt(sample_repo, mock_console):
    chat = InteractiveChat(repo_path=str(sample_repo), offline=True, console=mock_console)
    with patch.object(chat.console, "input", side_effect=KeyboardInterrupt):
        chat.start()
    output = mock_console.file.getvalue()
    assert "Exiting DocDrift chat session" in output

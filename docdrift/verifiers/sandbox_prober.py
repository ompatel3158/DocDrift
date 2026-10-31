"""Safe sandbox command prober for DocDrift."""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from docdrift.models import Claim, ClaimType, Verdict, VerificationResult
from docdrift.verifiers.base import BaseVerifier


# Blacklisted dangerous tokens and commands that are strictly rejected
DANGEROUS_EXACT_PATTERNS: List[re.Pattern] = [
    re.compile(r"\brm\b", re.IGNORECASE),
    re.compile(r"\bdel\b", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\bdrop\b", re.IGNORECASE),
    re.compile(r"\btruncate\b", re.IGNORECASE),
    re.compile(r"\bkill\b", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bchmod\b", re.IGNORECASE),
    re.compile(r"curl\s*(?:-[a-zA-Z]+\s+)*[^\n|]+\|\s*(?:ba)?sh", re.IGNORECASE),
    re.compile(r"wget\s*(?:-[a-zA-Z]+\s+)*[^\n|]+\|\s*(?:ba)?sh", re.IGNORECASE),
]

BLACKLISTED_SUBSTRINGS: List[str] = [
    "curl | sh",
    "wget | bash",
    "curl | bash",
    "wget | sh",
]


class ProbeResult(BaseModel):
    """Result of a safe sandbox command probe."""
    success: bool
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    command: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    is_safe: bool = True
    timed_out: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SandboxCommandProber(BaseVerifier):
    """Safe command prober that verifies CLI commands, subcommands, and flags.

    Capabilities:
    1. PATH toolchain verification (e.g. npm, docker, make, cargo, python, uv).
    2. Safe dry-run execution:
       - 'make -n <target>' with timeout.
       - '<bin> [subcommand] --help' with timeout.
    3. Strict security guards:
       - Rejects blacklisted dangerous tokens: rm, del, format, drop, truncate, kill, sudo, chmod, curl | sh, wget | bash.
       - Enforces strict execution timeout (default 3000ms).
    """

    DEFAULT_TIMEOUT_SEC: float = 3.0  # 3000 ms

    def can_verify(self, claim: Claim) -> bool:
        """Determines if this prober can verify the claim."""
        return claim.claim_type in (ClaimType.CLI_FLAG, ClaimType.COMMAND_SCRIPT)

    def verify(self, claim: Claim, repo_path: str) -> VerificationResult:
        """Verifies CLI flag claims or performs dynamic dry-run checks."""
        repo = Path(repo_path).resolve()

        if claim.claim_type == ClaimType.CLI_FLAG:
            return self._verify_cli_flag(claim)

        if claim.claim_type == ClaimType.COMMAND_SCRIPT:
            # Check for make dry-run opportunity
            cmd_str = claim.extracted_value.strip().strip("`")
            tokens = self._tokenize(cmd_str)
            if tokens and tokens[0] == "make":
                target = tokens[1] if len(tokens) > 1 and not tokens[1].startswith("-") else ""
                probe_res = self.probe_make_target(target, str(repo))
                if probe_res.success:
                    return VerificationResult(
                        verdict=Verdict.VERIFIED,
                        evidence=f"Dynamic dry-run 'make -n {target}' succeeded.",
                        suggested_value=None,
                        metadata={"stdout": probe_res.stdout},
                    )
                elif not probe_res.is_safe:
                    return VerificationResult(
                        verdict=Verdict.DRIFTED,
                        evidence=f"Command rejected: {probe_res.error}",
                        suggested_value=None,
                    )
                else:
                    return VerificationResult(
                        verdict=Verdict.DRIFTED,
                        evidence=f"Dry-run 'make -n {target}' failed: {probe_res.stderr or probe_res.error}",
                        suggested_value=None,
                    )

        return VerificationResult(
            verdict=Verdict.UNKNOWN,
            evidence=f"Sandbox prober could not deterministically probe '{claim.extracted_value}'.",
            suggested_value=None,
        )

    def probe_command(self, command_str: str, cwd: str) -> VerificationResult:
        """Safely probes a command via PATH toolchain checks or dry-run execution."""
        tokens = self._tokenize(command_str.strip().strip("`"))
        if not tokens:
            return VerificationResult(
                verdict=Verdict.UNKNOWN,
                evidence="Empty command string.",
            )
        binary = tokens[0]
        if not self.is_tool_available(binary):
            return VerificationResult(
                verdict=Verdict.UNKNOWN,
                evidence=f"Tool '{binary}' not currently installed on system PATH.",
            )
        # If make command, dry-run test
        if binary == "make":
            target = tokens[1] if len(tokens) > 1 and not tokens[1].startswith("-") else ""
            p_res = self.probe_make_target(target, cwd)
            if p_res.success:
                return VerificationResult(verdict=Verdict.VERIFIED, evidence="Dry-run 'make -n' verified on PATH.")
            else:
                return VerificationResult(verdict=Verdict.UNKNOWN, evidence=f"make dry-run: {p_res.stderr or p_res.error or 'unknown'}")

        return VerificationResult(
            verdict=Verdict.VERIFIED,
            evidence=f"Toolchain binary '{binary}' confirmed available on PATH.",
        )

    # --------------------------------------------------------------------------
    # Security Validation
    # --------------------------------------------------------------------------

    def is_safe_command(self, cmd: Union[str, List[str]]) -> Tuple[bool, Optional[str]]:
        """Strictly checks if a command is safe to execute.

        Returns (is_safe, error_reason).
        """
        if isinstance(cmd, list):
            cmd_str = " ".join(cmd)
        else:
            cmd_str = cmd

        # Check blacklisted literal substrings
        for bl in BLACKLISTED_SUBSTRINGS:
            if bl in cmd_str:
                return False, f"Command contains blacklisted phrase: '{bl}'"

        # Check dangerous regex patterns (word boundaries)
        for pattern in DANGEROUS_EXACT_PATTERNS:
            match = pattern.search(cmd_str)
            if match:
                return False, f"Command contains blacklisted token: '{match.group(0)}'"

        return True, None

    # --------------------------------------------------------------------------
    # Toolchain / PATH Inspection
    # --------------------------------------------------------------------------

    def is_tool_available(self, tool_name: str) -> bool:
        """Checks if a CLI tool/binary is available on the system PATH."""
        if not tool_name:
            return False
        # Clean any path separators
        clean_name = os.path.basename(tool_name.strip())
        return shutil.which(clean_name) is not None

    def check_toolchain(self, tools: List[str]) -> Dict[str, bool]:
        """Checks availability for a list of CLI tools on PATH."""
        return {tool: self.is_tool_available(tool) for tool in tools}

    # --------------------------------------------------------------------------
    # Safe Execution / Dry-runs
    # --------------------------------------------------------------------------

    def run_safe_command(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout_sec: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ProbeResult:
        """Executes a command strictly within security bounds and timeout limits."""
        timeout = timeout_sec if timeout_sec is not None else self.DEFAULT_TIMEOUT_SEC

        # 1. Security Check
        is_safe, reason = self.is_safe_command(cmd)
        if not is_safe:
            return ProbeResult(
                success=False,
                command=cmd,
                error=reason,
                is_safe=False,
            )

        # 2. Tool availability check
        binary = cmd[0]
        if not self.is_tool_available(binary):
            return ProbeResult(
                success=False,
                command=cmd,
                error=f"Binary '{binary}' not found on system PATH.",
                is_safe=True,
            )

        # 3. Execution with strict timeout
        try:
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)

            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=exec_env,
                check=False,
            )
            return ProbeResult(
                success=(proc.returncode == 0),
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                command=cmd,
                is_safe=True,
                timed_out=False,
            )

        except subprocess.TimeoutExpired:
            return ProbeResult(
                success=False,
                command=cmd,
                error=f"Command timed out after {int(timeout * 1000)}ms.",
                is_safe=True,
                timed_out=True,
            )
        except Exception as e:
            return ProbeResult(
                success=False,
                command=cmd,
                error=f"Execution error: {str(e)}",
                is_safe=True,
            )

    def probe_make_target(
        self,
        target: str,
        repo_path: str,
        timeout_ms: int = 3000,
    ) -> ProbeResult:
        """Executes 'make -n <target>' dry-run to test Makefile recipe validity."""
        timeout_sec = timeout_ms / 1000.0
        cmd = ["make", "-n"]
        if target:
            cmd.append(target)
        return self.run_safe_command(cmd, cwd=repo_path, timeout_sec=timeout_sec)

    def probe_cli_help(
        self,
        binary: str,
        subcommands: Optional[List[str]] = None,
        flag: Optional[str] = None,
        timeout_ms: int = 3000,
    ) -> ProbeResult:
        """Executes '<bin> [subcommands] --help' to verify flags and subcommands."""
        timeout_sec = timeout_ms / 1000.0
        cmd = [binary]
        if subcommands:
            cmd.extend(subcommands)
        cmd.append("--help")

        probe_res = self.run_safe_command(cmd, timeout_sec=timeout_sec)
        if not probe_res.success:
            return probe_res

        # If checking a specific flag in help output
        if flag:
            clean_flag = flag.strip()
            help_text = (probe_res.stdout + "\n" + probe_res.stderr)
            flag_found = clean_flag in help_text
            probe_res.metadata["flag_found"] = flag_found
            probe_res.metadata["flag"] = clean_flag
            if not flag_found:
                probe_res.success = False
                probe_res.error = f"Flag '{clean_flag}' not found in '--help' output."

        return probe_res

    # --------------------------------------------------------------------------
    # CLI Flag Verification
    # --------------------------------------------------------------------------

    def _verify_cli_flag(self, claim: Claim) -> VerificationResult:
        """Verifies CLI flag claims via --help output."""
        flag = claim.extracted_value.strip().strip("`")
        binary = claim.metadata.get("binary")
        subcommands = claim.metadata.get("subcommands", [])

        # Try to infer binary from raw_line if missing
        if not binary and claim.raw_line:
            tokens = self._tokenize(claim.raw_line.strip().lstrip("$#`"))
            if tokens:
                binary = tokens[0]
                if len(tokens) > 1 and not tokens[1].startswith("-"):
                    subcommands = [tokens[1]]

        if not binary:
            return VerificationResult(
                verdict=Verdict.UNKNOWN,
                evidence=f"Could not determine CLI binary for flag '{flag}'.",
                suggested_value=None,
            )

        if not self.is_tool_available(binary):
            return VerificationResult(
                verdict=Verdict.UNKNOWN,
                evidence=f"Binary '{binary}' is not available on system PATH to probe flag '{flag}'.",
                suggested_value=None,
            )

        probe_res = self.probe_cli_help(binary, subcommands=subcommands, flag=flag)
        cmd_display = " ".join([binary] + subcommands + ["--help"])
        if probe_res.success:
            return VerificationResult(
                verdict=Verdict.VERIFIED,
                evidence=f"CLI flag '{flag}' verified in '{cmd_display}' output.",
                suggested_value=None,
            )
        elif probe_res.metadata.get("flag_found") is False:
            return VerificationResult(
                verdict=Verdict.DRIFTED,
                evidence=f"CLI flag '{flag}' does not appear in '{cmd_display}'.",
                suggested_value=None,
            )

        return VerificationResult(
            verdict=Verdict.UNKNOWN,
            evidence=f"Could not verify flag '{flag}': {probe_res.error}",
            suggested_value=None,
        )

    @staticmethod
    def _tokenize(cmd_str: str) -> List[str]:
        """Safely tokenizes command string."""
        try:
            return shlex.split(cmd_str, posix=True)
        except ValueError:
            return cmd_str.split()

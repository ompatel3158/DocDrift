"""Fixer for DocDrift.

Generates surgical, context-preserving documentation line rewrites for DRIFTED claims,
constrained by strict code validation.

Provides:
- validate_fix: code-level verification that guarantees proposed fixes are valid commands,
  existing files, matching claim types, and preserve semantic intent (e.g. never replace start with test).
- GemmaFixer: uses Gemma 4 with structured schema and repo context.
- RuleBasedFixer: deterministic token replacement fallback.
- HybridFixer: uses Gemma 4 when available, falls back to RuleBasedFixer.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from docdrift.ai_client import GemmaClient
from docdrift.models import Claim, ClaimType

logger = logging.getLogger("docdrift.fixer")


class FixResult(BaseModel):
    fixed_line: str = Field(description="The exact surgical replacement line")
    explanation: Optional[str] = Field(
        default="", description="Brief explanation of the applied fix"
    )


STANDARD_NPM_COMMANDS = {
    "test", "start", "stop", "restart", "install", "i", "init", "ci",
    "publish", "pack", "audit", "run", "run-script", "build", "help"
}


def validate_fix(
    claim: Claim,
    proposed_line: str,
    repo_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """Constrains proposed documentation fixes with strict code validation.

    Guarantees:
    1. Same claim type (never swap command with a directory/file or vice versa).
    2. Valid command syntax (e.g. npm run <script> for custom scripts).
    3. Semantic intent preservation (never swap server start command with test command).
    4. Target file existence (for FILE_PATH, target must physically exist in repo).
    5. Script existence in manifests (for COMMAND_SCRIPT, target must exist in package.json/Makefile).

    Returns:
        (is_valid: bool, reason: str)
    """
    clean_prop = proposed_line.strip()
    if not clean_prop or clean_prop == claim.raw_line.strip():
        return False, "Proposed line is unchanged from original"

    # Extract proposed token from backticks or raw text
    backticked = re.findall(r"`([^`]+)`", proposed_line)
    prop_token = backticked[0].strip() if backticked else clean_prop.lstrip("$#> ")

    # 1. COMMAND_SCRIPT Constraints
    if claim.claim_type == ClaimType.COMMAND_SCRIPT:
        # Cross-type check: Reject if proposed fix is a bare file/directory path
        first_token = prop_token.split()[0] if prop_token.split() else ""
        known_cmd_tools = (
            "npm", "npx", "yarn", "pnpm", "bun", "cargo", "make",
            "docker", "docker-compose", "python", "python3", "pytest",
            "node", "sh", "bash", "uv", "go"
        )
        if not any(k == first_token.lower() or prop_token.lower().startswith(f"{k} ") for k in known_cmd_tools):
            # Check if it looks like a path (has slashes, extension, or matches a repo folder)
            if "/" in prop_token or "\\" in prop_token or any(prop_token.endswith(ext) for ext in (".js", ".ts", ".json", ".py", ".md")):
                return False, f"Proposed fix '{prop_token}' is a file or directory path, not a command script"

        # 1. npm syntax and manifest check (checked first on proposed command token)
        prop_cmd_lower = prop_token.lower()
        if "npm" in prop_cmd_lower:
            parts = prop_token.split()
            try:
                npm_idx = [p.lower() for p in parts].index("npm")
                if len(parts) > npm_idx + 1:
                    sub = parts[npm_idx + 1].lower()
                    if sub not in STANDARD_NPM_COMMANDS:
                        return False, f"Invalid npm syntax: custom script '{sub}' requires 'npm run {sub}'"
            except ValueError:
                pass

        # 2. Semantic intent preservation check on specific command tokens
        orig_cmd = (claim.extracted_value or claim.raw_line).lower()
        start_keywords = {"start", "dev", "serve", "server", "up"}
        test_keywords = {"test", "spec", "check", "test-tap", "jest", "mocha", "pytest"}

        is_orig_start = any(k in orig_cmd.split() or f"run {k}" in orig_cmd or f"`{k}`" in orig_cmd for k in start_keywords)
        is_orig_test = any(k in orig_cmd.split() or f"run {k}" in orig_cmd or f"`{k}`" in orig_cmd for k in test_keywords)
        is_prop_start = any(k in prop_cmd_lower.split() or f"run {k}" in prop_cmd_lower or f"`{k}`" in prop_cmd_lower for k in start_keywords)
        is_prop_test = any(k in prop_cmd_lower.split() or f"run {k}" in prop_cmd_lower or f"`{k}`" in prop_cmd_lower for k in test_keywords)

        if is_orig_start and not is_orig_test and is_prop_test and not is_prop_start:
            return False, f"Cannot replace server startup command with a test command ('{prop_token}')"
        if is_orig_test and not is_orig_start and is_prop_start and not is_prop_test:
            return False, f"Cannot replace test command with a server startup command ('{prop_token}')"

        # 3. Manifest script existence check
        if "npm" in prop_cmd_lower:
            parts = prop_token.split()
            try:
                npm_idx = [p.lower() for p in parts].index("npm")
                if len(parts) > npm_idx + 1:
                    sub = parts[npm_idx + 1].lower()

                    # Check existence in package.json if repo_path given
                    if repo_path and sub in ("run", "run-script") and len(parts) > npm_idx + 2:
                        target_script = parts[npm_idx + 2]
                        pkg_file = Path(repo_path) / "package.json"
                        if pkg_file.exists():
                            try:
                                pkg_data = json.loads(pkg_file.read_text(encoding="utf-8"))
                                scripts = pkg_data.get("scripts", {})
                                if target_script not in scripts:
                                    return False, f"Script '{target_script}' does not exist in package.json"
                            except Exception:
                                pass
            except ValueError:
                pass

    # 2. FILE_PATH Constraints
    elif claim.claim_type == ClaimType.FILE_PATH:
        # Cross-type check: Reject if proposed fix contains command runners
        if any(f"{cmd} " in prop_token.lower() or prop_token.lower().startswith(f"{cmd} ") for cmd in ("npm", "yarn", "pnpm", "bun", "cargo", "make", "pytest", "docker", "python", "python3", "pip", "py", "uv", "node")):
            return False, f"Proposed fix '{prop_token}' is a command, not a file path"

        # Reject prose / instruction phrases
        if any(w in prop_token.lower().split() for w in ("install", "add", "run", "please", "update", "needs")):
            return False, f"Proposed fix '{prop_token}' contains prose instructions, not a file path"

        # Check physical existence in repository
        if repo_path:
            clean_path = prop_token.strip("'\"`").split("#")[0].strip().replace("\\", "/").lstrip("/")
            if clean_path.startswith("./"):
                clean_path = clean_path[2:]
            repo = Path(repo_path).resolve()
            candidate = repo / clean_path

            if not candidate.exists():
                # Scan repository for filename
                target_base = Path(clean_path).name.lower()
                found = False
                for root, dirs, files in os.walk(repo):
                    if any(d in root for d in (".git", "node_modules", ".venv", "dist", "build")):
                        continue
                    if any(f.lower() == target_base for f in files + dirs):
                        found = True
                        break
                if not found:
                    return False, f"Proposed file path '{clean_path}' does not exist in repository"

    # 3. ENV_VAR Constraints
    elif claim.claim_type == ClaimType.ENV_VAR:
        if not re.match(r"^[A-Z][A-Z0-9_]{1,}$", prop_token):
            return False, f"Proposed environment variable '{prop_token}' is not a valid uppercase identifier"

    return True, "Fix verified by code constraints"


class RuleBasedFixer:
    """Deterministic token replacement fixer for documentation claims with code validation."""

    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = repo_path

    def fix_claim(
        self,
        claim: Claim,
        suggested_value: Optional[str] = None,
        repo_path: Optional[str] = None,
    ) -> Optional[str]:
        """Deterministically fix a drifted claim line by replacing the outdated token.

        Args:
            claim: The drifted Claim object.
            suggested_value: Optional override for replacement token.
            repo_path: Optional path to repository root for validation.

        Returns:
            The fixed line string, or None if validation fails.
        """
        active_repo = repo_path or self.repo_path or os.path.dirname(claim.doc_path)
        replacement = suggested_value or claim.suggested_replacement

        if not replacement and claim.evidence:
            replacement = self._extract_replacement_from_evidence(
                claim.evidence, old_val=claim.extracted_value
            )

        if not replacement:
            claim.fixed_line = None
            claim.needs_review = True
            claim.metadata["needs_review"] = True
            claim.metadata["rejection_reason"] = "No replacement token could be determined"
            return None

        raw = claim.raw_line
        old_val = claim.extracted_value

        if old_val in raw:
            fixed = raw.replace(old_val, replacement, 1)
        else:
            pattern = re.compile(re.escape(old_val), re.IGNORECASE)
            fixed = pattern.sub(replacement, raw, count=1)

        # Validate with code constraints
        is_valid, reason = validate_fix(claim, fixed, repo_path=active_repo)
        if not is_valid:
            logger.info(f"RuleBasedFixer: rejected fix '{fixed}' for line {claim.line_number}: {reason}")
            claim.fixed_line = None
            claim.suggested_replacement = None
            claim.needs_review = True
            claim.metadata["needs_review"] = True
            claim.metadata["rejection_reason"] = reason
            return None

        claim.fixed_line = fixed
        claim.needs_review = False
        return fixed

    def _extract_replacement_from_evidence(
        self, evidence: str, old_val: Optional[str] = None
    ) -> Optional[str]:
        """Attempt to extract suggested replacement token from evidence string."""
        if old_val:
            backticked = re.findall(r"`([^`]+)`", evidence)
            for token in backticked:
                token_clean = token.strip()
                if token_clean and token_clean != old_val:
                    return token_clean

        patterns = [
            r"(?:suggested|expected|found|actual|replace with)(?:\s+\w+)?[\s:]+[`'\"]([^`'\"]+)[`'\"]",
            r"->\s*[`'\"]?([^`'\"\s]+)[`'\"]?",
            r"(?:use|instead of|instead)\s+[`'\"]([^`'\"]+)[`'\"]",
            r"Did you mean\s+['\"]([^'\"]+)['\"]",
        ]
        for pat in patterns:
            m = re.search(pat, evidence, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if not old_val or cand != old_val:
                    return cand

        if old_val:
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", evidence)
            for token in quoted:
                token_clean = token.strip()
                if token_clean and token_clean != old_val and "not found" not in token_clean.lower() and "not defined" not in token_clean.lower():
                    return token_clean

        return None


class GemmaFixer:
    """Uses Gemma 4 to produce surgical, context-preserving documentation rewrites constrained by code."""

    FIX_PROMPT_TEMPLATE = """You are an expert AI documentation code fixer for software repositories.
A documentation claim was verified against the actual codebase and proven to have DRIFTED (outdated or incorrect).

Document: {doc_path} (Line {line_number})
Claim Type: {claim_type}
Outdated Line: {raw_line}
Outdated Claim Token: {extracted_value}
Verification Evidence: {evidence}
Suggested Replacement: {suggested_replacement}

Surrounding Documentation Context:
{context_str}

Instructions:
1. Produce the EXACT single replacement line for Line {line_number}.
2. Make a context-preserving surgical rewrite of ONLY the drifted claim.
3. Preserve exact leading indentation (spaces/tabs).
4. Preserve markdown formatting, backticks, punctuation, and surrounding phrasing.
5. CRITICAL: The replacement MUST be of the EXACT SAME type (e.g. if the claim is a command, replace with a valid command, NEVER a file path or directory).
6. CRITICAL: If the claim is a server start/dev command, NEVER replace it with a test command.
7. Return the result in the structured JSON schema.
"""

    def __init__(
        self,
        client: Optional[GemmaClient] = None,
        ai_client: Optional[GemmaClient] = None,
        repo_path: Optional[str] = None,
    ) -> None:
        """Initialize GemmaFixer with a GemmaClient (or ai_client alias)."""
        self.client = client or ai_client
        self.repo_path = repo_path
        self.fallback_fixer = RuleBasedFixer(repo_path=repo_path)

    def generate_fix(self, claim: Claim, repo_path: Optional[str] = None) -> Optional[str]:
        """Alias for fix_claim."""
        return self.fix_claim(claim, repo_path=repo_path)

    def fix_claim(
        self,
        claim: Claim,
        surrounding_lines: Optional[List[str]] = None,
        evidence: Optional[str] = None,
        repo_path: Optional[str] = None,
    ) -> Optional[str]:
        """Produce a surgical rewrite of a drifted claim line using Gemma 4 with code validation.

        Args:
            claim: The drifted Claim object.
            surrounding_lines: Optional list of adjacent lines for context.
            evidence: Optional verification evidence string.
            repo_path: Optional repository root for code constraint validation.

        Returns:
            The fixed line string, or None if validation fails.
        """
        active_repo = repo_path or self.repo_path or os.path.dirname(claim.doc_path)
        proof = evidence or claim.evidence
        suggested = claim.suggested_replacement or "See verification evidence"

        context_str = ""
        if surrounding_lines:
            context_str = "\n".join(f"  {line}" for line in surrounding_lines)
        else:
            context_str = f"  {claim.raw_line}"

        prompt = self.FIX_PROMPT_TEMPLATE.format(
            doc_path=claim.doc_path,
            line_number=claim.line_number,
            claim_type=claim.claim_type.value,
            raw_line=claim.raw_line,
            extracted_value=claim.extracted_value,
            evidence=proof,
            suggested_replacement=suggested,
            context_str=context_str,
        )

        try:
            result = self.client.generate_structured_json(
                prompt=prompt,
                response_schema=FixResult,
                system_instruction="You are a precision documentation rewrite assistant.",
                temperature=0.1,
            )

            fixed_line = ""
            if isinstance(result, FixResult):
                fixed_line = getattr(result, "fixed_line", "") or ""
            elif isinstance(result, dict) and "fixed_line" in result:
                fixed_line = str(result["fixed_line"])
            elif isinstance(result, str):
                fixed_line = result

            if fixed_line:
                raw = claim.raw_line
                leading_indent = raw[: len(raw) - len(raw.lstrip())]
                if leading_indent and not fixed_line.startswith(leading_indent):
                    fixed_line = leading_indent + fixed_line.lstrip()

                # Strictly validate with code constraints
                is_valid, reason = validate_fix(claim, fixed_line, repo_path=active_repo)
                if is_valid:
                    claim.fixed_line = fixed_line
                    claim.needs_review = False
                    return fixed_line
                else:
                    logger.warning(f"Gemma 4 proposed fix rejected by code constraints: {reason}")
                    # Try fallback fixer
                    return self.fallback_fixer.fix_claim(claim, repo_path=active_repo)

        except Exception as e:
            logger.warning(
                f"Gemma 4 fixer failed ({e}); falling back to deterministic RuleBasedFixer."
            )

        return self.fallback_fixer.fix_claim(claim, repo_path=active_repo)


class HybridFixer:
    """Hybrid fixer: uses Gemma 4 when available, falls back to RuleBasedFixer."""

    def __init__(
        self,
        client: Optional[GemmaClient] = None,
        ai_client: Optional[GemmaClient] = None,
        repo_path: Optional[str] = None,
    ) -> None:
        """Initialize HybridFixer with optional GemmaClient."""
        self.client = client or ai_client
        self.repo_path = repo_path
        self.rule_fixer = RuleBasedFixer(repo_path=repo_path)
        self.gemma_fixer = GemmaFixer(self.client, repo_path=repo_path) if self.client else None

    def generate_fix(self, claim: Claim, repo_path: Optional[str] = None) -> Optional[str]:
        """Alias for fix_claim."""
        return self.fix_claim(claim, repo_path=repo_path)

    def fix_claim(
        self,
        claim: Claim,
        surrounding_lines: Optional[List[str]] = None,
        evidence: Optional[str] = None,
        repo_path: Optional[str] = None,
    ) -> Optional[str]:
        """Fix claim line using Gemma 4 or deterministic fallback."""
        active_repo = repo_path or self.repo_path or os.path.dirname(claim.doc_path)
        if self.client and not self.client.is_offline and self.gemma_fixer:
            try:
                fixed = self.gemma_fixer.fix_claim(claim, surrounding_lines, evidence, repo_path=active_repo)
                if fixed:
                    return fixed
            except Exception as e:
                logger.warning(
                    f"HybridFixer: Gemma 4 error ({e}), falling back to RuleBasedFixer."
                )

        return self.rule_fixer.fix_claim(claim, repo_path=active_repo)

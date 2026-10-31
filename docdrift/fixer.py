"""Fixer for DocDrift.

Generates surgical, context-preserving documentation line rewrites for DRIFTED claims.

Provides:
- GemmaFixer: uses Gemma 4 with structured schema and repo context.
- RuleBasedFixer: deterministic token replacement fallback.
- HybridFixer: uses Gemma 4 when available, falls back to RuleBasedFixer.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from docdrift.ai_client import GemmaClient
from docdrift.models import Claim

logger = logging.getLogger("docdrift.fixer")


class FixResult(BaseModel):
    fixed_line: str = Field(description="The exact surgical replacement line")
    explanation: Optional[str] = Field(
        default="", description="Brief explanation of the applied fix"
    )


class RuleBasedFixer:
    """Deterministic token replacement fixer for documentation claims."""

    def fix_claim(
        self,
        claim: Claim,
        suggested_value: Optional[str] = None,
    ) -> str:
        """Deterministically fix a drifted claim line by replacing the outdated token.

        Args:
            claim: The drifted Claim object.
            suggested_value: Optional override for replacement token.

        Returns:
            The fixed line string.
        """
        replacement = suggested_value or claim.suggested_replacement

        # If not directly provided, check if evidence specifies a replacement
        if not replacement and claim.evidence:
            replacement = self._extract_replacement_from_evidence(
                claim.evidence, old_val=claim.extracted_value
            )

        if not replacement:
            claim.fixed_line = claim.raw_line
            return claim.raw_line

        raw = claim.raw_line
        old_val = claim.extracted_value

        if old_val in raw:
            fixed = raw.replace(old_val, replacement, 1)
        else:
            # Try case-insensitive or regex boundary match
            pattern = re.compile(re.escape(old_val), re.IGNORECASE)
            fixed = pattern.sub(replacement, raw, count=1)

        claim.fixed_line = fixed
        return fixed

    def _extract_replacement_from_evidence(
        self, evidence: str, old_val: Optional[str] = None
    ) -> Optional[str]:
        """Attempt to extract suggested replacement token from evidence string."""
        # 1. If evidence contains a backticked token different from old_val, prefer it
        if old_val:
            backticked = re.findall(r"`([^`]+)`", evidence)
            for token in backticked:
                token_clean = token.strip()
                if token_clean and token_clean != old_val:
                    return token_clean

        # 2. Check patterns like "actual file: `foo`", "expected: 'foo'", "-> `foo`"
        patterns = [
            r"(?:suggested|expected|found|actual|replace with)(?:\s+\w+)?[\s:]+[`'\"]([^`'\"]+)[`'\"]",
            r"->\s*[`'\"]?([^`'\"\s]+)[`'\"]?",
            r"(?:use|instead of|instead)\s+[`'\"]([^`'\"]+)[`'\"]",
        ]
        for pat in patterns:
            m = re.search(pat, evidence, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if not old_val or cand != old_val:
                    return cand

        # 3. Check any quoted token different from old_val
        if old_val:
            quoted = re.findall(r"['\"]([^'\"]+)['\"]", evidence)
            for token in quoted:
                token_clean = token.strip()
                if token_clean and token_clean != old_val and "not found" not in token_clean.lower():
                    return token_clean

        return None


class GemmaFixer:
    """Uses Gemma 4 to produce surgical, context-preserving documentation rewrites."""

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
5. Return the result in the structured JSON schema.
"""

    def __init__(
        self,
        client: Optional[GemmaClient] = None,
        ai_client: Optional[GemmaClient] = None,
    ) -> None:
        """Initialize GemmaFixer with a GemmaClient (or ai_client alias)."""
        self.client = client or ai_client
        self.fallback_fixer = RuleBasedFixer()

    def generate_fix(self, claim: Claim) -> str:
        """Alias for fix_claim."""
        return self.fix_claim(claim)

    def fix_claim(
        self,
        claim: Claim,
        surrounding_lines: Optional[List[str]] = None,
        evidence: Optional[str] = None,
    ) -> str:
        """Produce a surgical rewrite of a drifted claim line using Gemma 4.

        Args:
            claim: The drifted Claim object.
            surrounding_lines: Optional list of adjacent lines for context.
            evidence: Optional verification evidence string.

        Returns:
            The fixed line string.
        """
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
                # Preserve leading indentation if missing
                raw = claim.raw_line
                leading_indent = raw[: len(raw) - len(raw.lstrip())]
                if leading_indent and not fixed_line.startswith(leading_indent):
                    fixed_line = leading_indent + fixed_line.lstrip()

                claim.fixed_line = fixed_line
                return fixed_line

        except Exception as e:
            logger.warning(
                f"Gemma 4 fixer failed ({e}); falling back to deterministic RuleBasedFixer."
            )

        return self.fallback_fixer.fix_claim(claim)


class HybridFixer:
    """Hybrid fixer: uses Gemma 4 when available, falls back to RuleBasedFixer."""

    def __init__(
        self,
        client: Optional[GemmaClient] = None,
        ai_client: Optional[GemmaClient] = None,
    ) -> None:
        """Initialize HybridFixer with optional GemmaClient."""
        self.client = client or ai_client
        self.rule_fixer = RuleBasedFixer()
        self.gemma_fixer = GemmaFixer(self.client) if self.client else None

    def generate_fix(self, claim: Claim) -> str:
        """Alias for fix_claim."""
        return self.fix_claim(claim)

    def fix_claim(
        self,
        claim: Claim,
        surrounding_lines: Optional[List[str]] = None,
        evidence: Optional[str] = None,
    ) -> str:
        """Fix claim line using Gemma 4 or deterministic fallback.

        Args:
            claim: The drifted Claim object.
            surrounding_lines: Optional surrounding lines context.
            evidence: Optional verification evidence string.

        Returns:
            The fixed line string.
        """
        if self.client and not self.client.is_offline and self.gemma_fixer:
            try:
                fixed = self.gemma_fixer.fix_claim(claim, surrounding_lines, evidence)
                if fixed:
                    return fixed
            except Exception as e:
                logger.warning(
                    f"HybridFixer: Gemma 4 error ({e}), falling back to RuleBasedFixer."
                )

        return self.rule_fixer.fix_claim(claim)

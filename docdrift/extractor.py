"""Claim Extractor for DocDrift.

Extracts checkable documentation claims (file paths, command scripts,
environment variables, docker ports) from DocSection instances.

Provides:
- GemmaClaimExtractor: queries Gemma 4 with structured schema.
- RuleBasedExtractor: deterministic regex-based extraction.
- HybridExtractor: seamlessly uses Gemma 4 when available, falls back to RuleBasedExtractor.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from docdrift.ai_client import GemmaClient
from docdrift.models import Claim, ClaimType, DocSection

logger = logging.getLogger("docdrift.extractor")


# Pydantic schema for structured Gemma 4 claim extraction
class ExtractedClaimItem(BaseModel):
    line_number: int = Field(description="1-based line number in document where claim appears")
    raw_line: str = Field(description="Exact line text containing the claim")
    claim_type: str = Field(
        description="One of: 'file_path', 'command_script', 'env_var', 'docker_port'"
    )
    extracted_value: str = Field(
        description="The specific claim token (e.g. 'src/app.py', 'npm test', 'PORT', '8080:8080')"
    )


class ExtractionResponse(BaseModel):
    claims: List[ExtractedClaimItem] = Field(default_factory=list)


# Known file extensions for deterministic path identification
FILE_EXTENSIONS = {
    ".py", ".ts", ".js", ".jsx", ".tsx", ".json", ".yaml", ".yml",
    ".toml", ".md", ".sh", ".bash", ".zsh", ".rs", ".go", ".java",
    ".kt", ".c", ".cpp", ".h", ".hpp", ".cs", ".html", ".css",
    ".scss", ".sql", ".env", ".lock", ".xml", ".ini", ".cfg", ".conf",
}

# Standalone recognized filenames without extensions
SPECIAL_FILENAMES = {
    "dockerfile", "makefile", "procfile", "gemfile", "vagrantfile",
    "jenkinsfile", ".gitignore", ".env.example", ".env.local", "license",
}

# Known command prefixes/utilities
COMMAND_PREFIXES = (
    "npm ", "npx ", "yarn ", "pnpm ", "cargo ", "make ", "docker ",
    "docker-compose ", "pytest", "python -m", "pip install", "uv ",
    "go run", "go test", "go build", "mvn ", "gradle ",
)

# Common single-word env vars without underscores
COMMON_ENV_VARS = {"PORT", "HOST", "DEBUG", "SECRET", "TOKEN", "SALT", "NODE_ENV"}


class RuleBasedExtractor:
    """Deterministic regex-based documentation claim extractor."""

    def extract(self, section: DocSection) -> List[Claim]:
        """Extract claims deterministically from a DocSection.

        Args:
            section: DocSection containing lines and document metadata.

        Returns:
            List of Claim instances.
        """
        claims: List[Claim] = []
        seen: Set[Tuple[int, ClaimType, str]] = set()

        for idx, line in enumerate(section.lines):
            line_no = section.start_line + idx
            line_str = line.strip()
            if not line_str:
                continue

            # 1. Backticked tokens extraction
            backticked = re.findall(r"`([^`]+)`", line)
            for token in backticked:
                token_clean = token.strip()
                if not token_clean:
                    continue

                # Check: Command Script
                if self._is_command(token_clean):
                    self._add_claim(
                        claims, seen, section, line_no, line,
                        ClaimType.COMMAND_SCRIPT, token_clean
                    )
                    continue

                # Check: File Path
                if self._is_file_path(token_clean):
                    self._add_claim(
                        claims, seen, section, line_no, line,
                        ClaimType.FILE_PATH, token_clean
                    )
                    continue

                # Check: Docker Port (e.g. `8080:8080`, `3000:3000`)
                port_match = re.match(r"^(\d{2,5}:\d{2,5})$", token_clean)
                if port_match:
                    self._add_claim(
                        claims, seen, section, line_no, line,
                        ClaimType.DOCKER_PORT, port_match.group(1)
                    )
                    continue

                # Check: Environment Variable
                if self._is_env_var(token_clean):
                    self._add_claim(
                        claims, seen, section, line_no, line,
                        ClaimType.ENV_VAR, token_clean
                    )
                    continue

            # 2. Line-level command extraction (e.g. $ npm test, or line starting with command)
            self._extract_line_commands(claims, seen, section, line_no, line)

            # 3. Line-level Docker Port extraction (e.g. -p 8080:8080, EXPOSE 3000, localhost:5000)
            self._extract_line_ports(claims, seen, section, line_no, line)

            # 4. Line-level Env Var extraction (e.g. export FOO=bar, DATABASE_URL=postgres://...)
            self._extract_line_env_vars(claims, seen, section, line_no, line)

        return claims

    def _is_file_path(self, token: str) -> bool:
        """Check if token represents a file or directory path."""
        # Exclude URLs and commands
        if re.match(r"^[a-zA-Z]+://", token) or " " in token:
            return False
        # Exclude CLI options like --version, -rf
        if token.startswith("-"):
            return False

        lower = token.lower()

        # Check special filenames
        base = lower.split("/")[-1]
        if base in SPECIAL_FILENAMES:
            return True

        # Check path slashes
        if "/" in token or "\\" in token:
            # Avoid mathematical or generic expressions
            if re.search(r"[a-zA-Z0-9_\-\.]+[/\\][a-zA-Z0-9_\-\.]+", token):
                return True

        # Check extension
        for ext in FILE_EXTENSIONS:
            if lower.endswith(ext) and len(lower) > len(ext):
                return True

        return False

    def _is_command(self, token: str) -> bool:
        """Check if token represents a runnable command or script."""
        for prefix in COMMAND_PREFIXES:
            if token.startswith(prefix) or token == prefix.strip():
                return True
        if token.startswith("./") and (" " in token or token.endswith(".sh") or token.endswith(".py")):
            return True
        return False

    def _is_env_var(self, token: str) -> bool:
        """Check if token is an environment variable name."""
        if not re.match(r"^[A-Z][A-Z0-9_]{1,}$", token):
            return False
        # Must have an underscore (e.g. DATABASE_URL) or be a recognized standard env var
        if "_" in token or token in COMMON_ENV_VARS:
            return True
        return False

    def _extract_line_commands(
        self,
        claims: List[Claim],
        seen: Set[Tuple[int, ClaimType, str]],
        section: DocSection,
        line_no: int,
        line: str,
    ) -> None:
        stripped = line.strip()
        # Prompt prefix removal: "$ npm test" -> "npm test"
        cmd_candidate = re.sub(r"^[$#>]\s*", "", stripped)
        for prefix in COMMAND_PREFIXES:
            if cmd_candidate.startswith(prefix):
                self._add_claim(
                    claims, seen, section, line_no, line,
                    ClaimType.COMMAND_SCRIPT, cmd_candidate
                )
                break

    def _extract_line_ports(
        self,
        claims: List[Claim],
        seen: Set[Tuple[int, ClaimType, str]],
        section: DocSection,
        line_no: int,
        line: str,
    ) -> None:
        # Match -p 8080:8080 or --publish 8080:8080
        docker_port_match = re.search(r"(?:-p|--publish)\s+[\"']?(\d{2,5}:\d{2,5})[\"']?", line)
        if docker_port_match:
            self._add_claim(
                claims, seen, section, line_no, line,
                ClaimType.DOCKER_PORT, docker_port_match.group(1)
            )
            return

        # Match EXPOSE 8080
        expose_match = re.search(r"\bEXPOSE\s+(\d{2,5})\b", line, re.IGNORECASE)
        if expose_match:
            self._add_claim(
                claims, seen, section, line_no, line,
                ClaimType.DOCKER_PORT, expose_match.group(1)
            )
            return

        # Match localhost:3000 or 127.0.0.1:8000
        host_port = re.search(r"(?:localhost|127\.0\.0\.1):(\d{2,5})", line)
        if host_port:
            self._add_claim(
                claims, seen, section, line_no, line,
                ClaimType.DOCKER_PORT, host_port.group(1)
            )

    def _extract_line_env_vars(
        self,
        claims: List[Claim],
        seen: Set[Tuple[int, ClaimType, str]],
        section: DocSection,
        line_no: int,
        line: str,
    ) -> None:
        # Match export FOO_BAR=...
        export_match = re.findall(r"\bexport\s+([A-Z][A-Z0-9_]{2,})=", line)
        for var in export_match:
            self._add_claim(
                claims, seen, section, line_no, line,
                ClaimType.ENV_VAR, var
            )

        # Match inline VAR=val assignment if VAR is uppercase
        assign_matches = re.findall(r"\b([A-Z][A-Z0-9_]{2,})=(?!\s)", line)
        for var in assign_matches:
            if "_" in var or var in COMMON_ENV_VARS:
                self._add_claim(
                    claims, seen, section, line_no, line,
                    ClaimType.ENV_VAR, var
                )

    def _add_claim(
        self,
        claims: List[Claim],
        seen: Set[Tuple[int, ClaimType, str]],
        section: DocSection,
        line_no: int,
        raw_line: str,
        claim_type: ClaimType,
        extracted_value: str,
    ) -> None:
        key = (line_no, claim_type, extracted_value)
        if key in seen:
            return
        seen.add(key)
        claims.append(
            Claim(
                doc_path=section.doc_path,
                line_number=line_no,
                raw_line=raw_line,
                section_title=section.section_title,
                claim_type=claim_type,
                extracted_value=extracted_value,
            )
        )


class GemmaClaimExtractor:
    """Uses Gemma 4 structured JSON generation to accurately extract claims."""

    EXTRACTION_PROMPT_TEMPLATE = """You are an expert documentation code verifier.
Analyze the following documentation section and extract all checkable technical claims.

Documentation File: {doc_path}
Section Title: {section_title}

Section Content (with line numbers):
{numbered_lines}

Allowed Claim Types:
- 'file_path': Source file paths, config files, assets mentioned (e.g., `src/index.ts`, `pyproject.toml`, `Dockerfile`).
- 'command_script': Build, test, run, or CLI package manager commands (e.g., `npm run build`, `cargo test`, `docker build`).
- 'env_var': Environment variables and configuration keys (e.g., `DATABASE_URL`, `PORT`, `API_KEY`).
- 'docker_port': Port numbers or docker port mappings (e.g., `8080:8080`, `localhost:3000`, `EXPOSE 5000`).

Extract each claim with:
- line_number: The integer line number where the claim appears.
- raw_line: The exact line text from the snippet.
- claim_type: Must be one of 'file_path', 'command_script', 'env_var', 'docker_port'.
- extracted_value: The specific token/value to verify against code.
"""

    def __init__(self, client: GemmaClient) -> None:
        """Initialize GemmaClaimExtractor with a GemmaClient."""
        self.client = client

    def extract(self, section: DocSection) -> List[Claim]:
        """Extract claims from DocSection using Gemma 4.

        Args:
            section: DocSection to extract claims from.

        Returns:
            List of Claim instances.
        """
        numbered_lines = "\n".join(
            f"{section.start_line + i}: {line}"
            for i, line in enumerate(section.lines)
        )

        prompt = self.EXTRACTION_PROMPT_TEMPLATE.format(
            doc_path=section.doc_path,
            section_title=section.section_title,
            numbered_lines=numbered_lines,
        )

        try:
            response = self.client.generate_structured_json(
                prompt=prompt,
                response_schema=ExtractionResponse,
                system_instruction="You are a precise technical claim extraction engine for software documentation.",
                temperature=0.1,
            )
        except Exception as e:
            logger.error(f"Gemma 4 claim extraction failed: {e}")
            raise

        raw_items: List[ExtractedClaimItem] = []
        if isinstance(response, ExtractionResponse):
            raw_items = response.claims
        elif isinstance(response, dict) and "claims" in response:
            for item in response["claims"]:
                if isinstance(item, dict):
                    raw_items.append(ExtractedClaimItem(**item))
        elif isinstance(response, list):
            for item in response:
                if isinstance(item, dict):
                    raw_items.append(ExtractedClaimItem(**item))

        claims: List[Claim] = []
        for item in raw_items:
            try:
                c_type = ClaimType(item.claim_type.lower())
            except ValueError:
                # Skip invalid or unrecognized types
                continue

            # Verify and resolve line number
            target_line_no = item.line_number
            idx = target_line_no - section.start_line
            actual_line = item.raw_line
            if 0 <= idx < len(section.lines):
                actual_line = section.lines[idx]

            claims.append(
                Claim(
                    doc_path=section.doc_path,
                    line_number=target_line_no,
                    raw_line=actual_line,
                    section_title=section.section_title,
                    claim_type=c_type,
                    extracted_value=item.extracted_value,
                )
            )

        return claims


class HybridExtractor:
    """Hybrid extractor: uses Gemma 4 when available, falls back to RuleBasedExtractor."""

    def __init__(
        self,
        client: Optional[GemmaClient] = None,
        ai_client: Optional[GemmaClient] = None,
        memory: Optional[Any] = None,
    ) -> None:
        """Initialize HybridExtractor.

        Args:
            client: Optional GemmaClient. If None or offline, uses RuleBasedExtractor.
            ai_client: Alias for client.
            memory: Optional memory backend.
        """
        self.client = client or ai_client
        self.memory = memory
        self.rule_extractor = RuleBasedExtractor()
        self.gemma_extractor = GemmaClaimExtractor(self.client) if self.client else None

    def extract(self, section: DocSection) -> List[Claim]:
        """Extract claims, preferring Gemma 4 and falling back to rule-based extraction.

        Args:
            section: DocSection to inspect.

        Returns:
            List of Claim instances.
        """
        # If client is configured and online, attempt Gemma 4 extraction
        if self.client and not self.client.is_offline and self.gemma_extractor:
            try:
                gemma_claims = self.gemma_extractor.extract(section)
                if gemma_claims:
                    return gemma_claims
                logger.debug("Gemma 4 returned no claims; falling back to rule-based extractor.")
            except Exception as e:
                logger.warning(
                    f"Gemma 4 claim extraction encountered error ({e}); falling back to rule-based extractor."
                )

        # Seamless fallback
        return self.rule_extractor.extract(section)

    def extract_claims(self, section: DocSection) -> List[Claim]:
        """Alias for extract()."""
        return self.extract(section)


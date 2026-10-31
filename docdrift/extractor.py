"""Claim Extractor for DocDrift.

Extracts checkable documentation claims (file paths, command scripts,
environment variables, docker ports) from DocSection instances.

Provides:
- GemmaClaimExtractor: queries Gemma 4 with structured schema and execution context tagging (runs_in).
- RuleBasedExtractor: deterministic regex-based extraction with context awareness.
- HybridExtractor: seamlessly uses Gemma 4 when available, falls back to RuleBasedExtractor.
- apply_runs_in_safety_net: enforces execution context boundaries (repo vs generated_app vs user_machine vs external).
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
    runs_in: str = Field(
        default="repo",
        description="Execution context: 'repo' (in repository root/source), 'generated_app' (tutorial demo app, created via generator or cd), 'user_machine' (global tool, /tmp, ~, home dir, OS level), or 'external' (remote URL)"
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
    "docker-compose ", "pytest", "python ", "python3 ", "py ", "pip ", "uv ",
    "uvicorn ", "gunicorn ", "cd ", "cp ", "copy ", "copy-item ",
    "go run", "go test", "go build", "mvn ", "gradle ",
)

# Common single-word env vars without underscores
COMMON_ENV_VARS = {"PORT", "HOST", "DEBUG", "SECRET", "TOKEN", "SALT", "NODE_ENV"}


def apply_runs_in_safety_net(claims: List[Claim]) -> List[Claim]:
    """Safety net enforcing execution context boundaries on all claims.

    Guarantees:
    - Absolute OS paths (/tmp, /var, ~, Windows drive) -> 'user_machine'.
    - Global package manager commands (-g, --global) -> 'user_machine'.
    - Scaffolding commands (express <dir>, create-react-app, etc.) -> 'generated_app'.
    - Subsequent commands within a code block following a cd into a generated/tmp directory -> 'generated_app'.
    - External links/URLs -> 'external'.
    """
    for c in claims:
        val = c.extracted_value.strip().strip("'\"`")
        raw = c.raw_line or ""
        raw_lower = raw.lower()

        # 1. External URL check
        if re.match(r"^(https?://|ftp://)", val, re.IGNORECASE) or re.search(r"\]\((https?://)", raw):
            c.runs_in = "external"
            continue

        # 2. Absolute user-machine path check (/tmp, /var, /etc, ~, C:\, D:\)
        if (
            re.search(r"(?:^|[/\\])(tmp|var|etc|home)(?:[/\\]|$)", val, re.IGNORECASE)
            or val.startswith(("/tmp", "\\tmp", "~"))
            or re.match(r"^[A-Za-z]:[/\\]", val)
            or re.search(r"\b(/tmp/|\\tmp\\|~/)", raw)
        ):
            c.runs_in = "user_machine"
            continue

        # 3. Global tool install commands
        if any(g in raw_lower for g in ("-g ", "--global", "install -g", "pip install --user", "brew install", "apt-get")):
            c.runs_in = "user_machine"
            continue

        # 4. Scaffolding / generator commands
        if any(k in raw_lower for k in ("express /", "express .", "create-react-app", "npm create", "pnpm create", "yarn create", "cargo new", "go mod init", "flutter create", "django-admin")):
            c.runs_in = "generated_app"
            continue

        # 5. Commands following cd into non-repo directory
        if "cd " in raw_lower and any(p in raw_lower for p in ("/tmp", "~", "temp", "foo", "bar", "my-app", "myapp")):
            c.runs_in = "generated_app"
            continue

    return claims


class RuleBasedExtractor:
    """Deterministic regex-based documentation claim extractor with execution context awareness."""

    def extract(self, section: DocSection) -> List[Claim]:
        """Extract claims deterministically from a DocSection.

        Args:
            section: DocSection containing lines and document metadata.

        Returns:
            List of Claim instances.
        """
        claims: List[Claim] = []
        seen: Set[Tuple[int, ClaimType, str]] = set()

        in_code_block = False
        in_generated_context = False

        for idx, line in enumerate(section.lines):
            line_no = section.start_line + idx
            line_str = line.strip()

            if line_str.startswith("```"):
                if in_code_block:
                    in_code_block = False
                    in_generated_context = False
                else:
                    in_code_block = True
                continue

            if not line_str:
                if not in_code_block:
                    in_generated_context = False
                continue

            line_lower = line_str.lower()

            # Check if this line introduces a generated app or cd to ephemeral dir
            if any(k in line_lower for k in ("express /", "express .", "create-react-app", "npm create", "pnpm create", "yarn create", "cargo new", "go mod init", "flutter create", "django-admin")):
                in_generated_context = True
            elif "cd " in line_lower and any(p in line_lower for p in ("/tmp", "~", "temp", "foo", "bar", "my-app", "myapp")):
                in_generated_context = True

            # 1. Backticked tokens extraction
            backticked = re.findall(r"`([^`]+)`", line)
            for token in backticked:
                token_clean = token.strip()
                if not token_clean:
                    continue

                runs_in = self._determine_runs_in(line, token_clean, None, in_generated_context)

                # Check: Command Script
                if self._is_command(token_clean):
                    self._add_claim(
                        claims, seen, section, line_no, line,
                        ClaimType.COMMAND_SCRIPT, token_clean, runs_in=runs_in
                    )
                    continue

                # Check: File Path
                if self._is_file_path(token_clean):
                    self._add_claim(
                        claims, seen, section, line_no, line,
                        ClaimType.FILE_PATH, token_clean, runs_in=runs_in
                    )
                    continue

                # Check: Docker Port (e.g. `8080:8080`, `3000:3000`)
                port_match = re.match(r"^(\d{2,5}:\d{2,5})$", token_clean)
                if port_match:
                    self._add_claim(
                        claims, seen, section, line_no, line,
                        ClaimType.DOCKER_PORT, port_match.group(1), runs_in=runs_in
                    )
                    continue

                # Check: Environment Variable
                if self._is_env_var(token_clean):
                    self._add_claim(
                        claims, seen, section, line_no, line,
                        ClaimType.ENV_VAR, token_clean, runs_in=runs_in
                    )
                    continue

            # 2. Line-level command extraction (e.g. $ npm test, or line starting with command)
            self._extract_line_commands(claims, seen, section, line_no, line, in_generated_context)

            # 3. Line-level Docker Port extraction (e.g. -p 8080:8080, EXPOSE 3000, localhost:5000)
            self._extract_line_ports(claims, seen, section, line_no, line, in_generated_context)

            # 4. Line-level Env Var extraction (e.g. export FOO=bar, DATABASE_URL=postgres://...)
            self._extract_line_env_vars(claims, seen, section, line_no, line, in_generated_context)

        return apply_runs_in_safety_net(claims)

    def _determine_runs_in(
        self,
        line: str,
        token: str,
        claim_type: Optional[ClaimType],
        in_generated_context: bool,
    ) -> str:
        """Determines the execution context ('repo', 'generated_app', 'user_machine', 'external')."""
        token_clean = token.strip().strip("'\"`")
        line_lower = line.lower()

        # External URLs
        if re.match(r"^(https?://|ftp://)", token_clean, re.IGNORECASE) or re.search(r"\]\((https?://)", line):
            return "external"

        # User machine absolute paths
        if (
            re.search(r"(?:^|[/\\])(tmp|var|etc|home)(?:[/\\]|$)", token_clean, re.IGNORECASE)
            or token_clean.startswith(("/tmp", "\\tmp", "~"))
            or re.match(r"^[A-Za-z]:[/\\]", token_clean)
        ):
            return "user_machine"

        # Global tools
        if any(g in line_lower for g in ("-g ", "--global", "install -g", "pip install --user", "brew install", "apt-get")):
            return "user_machine"

        # Generator commands
        if any(k in line_lower for k in ("express /", "express .", "create-react-app", "npm create", "pnpm create", "yarn create", "cargo new", "go mod init", "flutter create", "django-admin")):
            return "generated_app"

        # Already inside generated context
        if in_generated_context:
            return "generated_app"

        return "repo"

    def _is_file_path(self, token: str) -> bool:
        """Check if token represents a file or directory path."""
        if re.match(r"^[a-zA-Z]+://", token) or " " in token:
            return False
        if token.startswith("-"):
            return False

        lower = token.lower()

        base = lower.split("/")[-1]
        if base in SPECIAL_FILENAMES:
            return True

        if "/" in token or "\\" in token:
            if re.search(r"[a-zA-Z0-9_\-\.]+[/\\][a-zA-Z0-9_\-\.]+", token):
                return True

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
        if token.startswith("express ") or token.startswith("git "):
            return True
        return False

    def _is_env_var(self, token: str) -> bool:
        """Check if token is an environment variable name."""
        if not re.match(r"^[A-Z][A-Z0-9_]{1,}$", token):
            return False
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
        in_generated_context: bool,
    ) -> None:
        stripped = line.strip()
        cmd_candidate = re.sub(r"^[$#>]\s*", "", stripped)
        runs_in = self._determine_runs_in(line, cmd_candidate, ClaimType.COMMAND_SCRIPT, in_generated_context)

        all_prefixes = COMMAND_PREFIXES + ("express ", "git ")
        for prefix in all_prefixes:
            if cmd_candidate.startswith(prefix):
                self._add_claim(
                    claims, seen, section, line_no, line,
                    ClaimType.COMMAND_SCRIPT, cmd_candidate, runs_in=runs_in
                )
                break

    def _extract_line_ports(
        self,
        claims: List[Claim],
        seen: Set[Tuple[int, ClaimType, str]],
        section: DocSection,
        line_no: int,
        line: str,
        in_generated_context: bool,
    ) -> None:
        runs_in = self._determine_runs_in(line, "", ClaimType.DOCKER_PORT, in_generated_context)
        docker_port_match = re.search(r"(?:-p|--publish)\s+[\"']?(\d{2,5}:\d{2,5})[\"']?", line)
        if docker_port_match:
            self._add_claim(
                claims, seen, section, line_no, line,
                ClaimType.DOCKER_PORT, docker_port_match.group(1), runs_in=runs_in
            )
            return

        expose_match = re.search(r"\bEXPOSE\s+(\d{2,5})\b", line, re.IGNORECASE)
        if expose_match:
            self._add_claim(
                claims, seen, section, line_no, line,
                ClaimType.DOCKER_PORT, expose_match.group(1), runs_in=runs_in
            )
            return

        host_port = re.search(r"(?:localhost|127\.0\.0\.1):(\d{2,5})", line)
        if host_port:
            self._add_claim(
                claims, seen, section, line_no, line,
                ClaimType.DOCKER_PORT, host_port.group(1), runs_in=runs_in
            )

    def _extract_line_env_vars(
        self,
        claims: List[Claim],
        seen: Set[Tuple[int, ClaimType, str]],
        section: DocSection,
        line_no: int,
        line: str,
        in_generated_context: bool,
    ) -> None:
        runs_in = self._determine_runs_in(line, "", ClaimType.ENV_VAR, in_generated_context)
        export_match = re.findall(r"\bexport\s+([A-Z][A-Z0-9_]{2,})=", line)
        for var in export_match:
            self._add_claim(
                claims, seen, section, line_no, line,
                ClaimType.ENV_VAR, var, runs_in=runs_in
            )

        assign_matches = re.findall(r"\b([A-Z][A-Z0-9_]{2,})=(?!\s)", line)
        for var in assign_matches:
            if "_" in var or var in COMMON_ENV_VARS:
                self._add_claim(
                    claims, seen, section, line_no, line,
                    ClaimType.ENV_VAR, var, runs_in=runs_in
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
        runs_in: str = "repo",
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
                runs_in=runs_in,
            )
        )


class GemmaClaimExtractor:
    """Uses Gemma 4 structured JSON generation to accurately extract claims and execution contexts."""

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

CRITICAL: Execution Context Tagging (`runs_in`):
For each claim, determine where it executes or belongs:
- 'repo': Commands, scripts, or files that belong directly to THIS repository and its subfolders/monorepo packages (e.g. `cd backend`, `cd frontend`, `npm test`, `uvicorn app.main:app`, `python -m pip install -r requirements.txt`). These MUST be verified against repository code and manifests.
- 'generated_app': Tutorial or Quick Start commands where the reader creates a brand-new demo project from scratch outside the repo using a generator (e.g. `express /tmp/foo && cd /tmp/foo`, then `npm start`, or `create-react-app myapp`). Do NOT tag existing repo directories (like `frontend/`, `backend/`, `cd backend`) as generated_app!
- 'user_machine': Commands installing global CLI tools (`npm install -g express-generator`, `brew install`, `apt-get`), or paths in `/tmp`, `/var`, `~`, or user home directories.
- 'external': External URLs, documentation links, or third-party web services.

Extract each claim with:
- line_number: The integer line number where the claim appears.
- raw_line: The exact line text from the snippet.
- claim_type: Must be one of 'file_path', 'command_script', 'env_var', 'docker_port'.
- extracted_value: The specific token/value to verify against code.
- runs_in: 'repo', 'generated_app', 'user_machine', or 'external'.
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
                continue

            target_line_no = item.line_number
            idx = target_line_no - section.start_line
            actual_line = item.raw_line
            if 0 <= idx < len(section.lines):
                actual_line = section.lines[idx]

            runs_in_val = getattr(item, "runs_in", "repo") or "repo"
            if runs_in_val.lower() not in ("repo", "generated_app", "user_machine", "external"):
                runs_in_val = "repo"

            claims.append(
                Claim(
                    doc_path=section.doc_path,
                    line_number=target_line_no,
                    raw_line=actual_line,
                    section_title=section.section_title,
                    claim_type=c_type,
                    extracted_value=item.extracted_value,
                    runs_in=runs_in_val.lower(),
                )
            )

        return apply_runs_in_safety_net(claims)


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
        if self.client and not self.client.is_offline and self.gemma_extractor:
            try:
                gemma_claims = self.gemma_extractor.extract(section)
                if gemma_claims:
                    return apply_runs_in_safety_net(gemma_claims)
                logger.debug("Gemma 4 returned no claims; falling back to rule-based extractor.")
            except Exception as e:
                logger.warning(
                    f"Gemma 4 claim extraction encountered error ({e}); falling back to rule-based extractor."
                )

        return apply_runs_in_safety_net(self.rule_extractor.extract(section))

    def extract_claims(self, section: DocSection) -> List[Claim]:
        """Alias for extract()."""
        return self.extract(section)

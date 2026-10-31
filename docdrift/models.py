from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class ClaimType(str, Enum):
    FILE_PATH = "file_path"
    COMMAND_SCRIPT = "command_script"
    ENV_VAR = "env_var"
    DOCKER_PORT = "docker_port"
    CLI_FLAG = "cli_flag"


class Verdict(str, Enum):
    VERIFIED = "verified"     # Code check confirmed the claim is accurate
    DRIFTED = "drifted"       # Code check proved the claim is outdated/false
    UNKNOWN = "unknown"       # Cannot be determined deterministically


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    doc_path: str
    line_number: int
    raw_line: str
    section_title: Optional[str] = None
    claim_type: ClaimType
    extracted_value: str
    verdict: Verdict = Verdict.UNKNOWN
    evidence: str = ""
    suggested_replacement: Optional[str] = None
    fixed_line: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocSection(BaseModel):
    doc_path: str
    section_title: str
    start_line: int
    lines: List[str]
    content_hash: str = ""

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class VerificationResult(BaseModel):
    verdict: Verdict
    evidence: str
    suggested_value: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DriftReport(BaseModel):
    repo_path: str
    files_scanned: List[str]
    total_claims: int
    verified_count: int
    drifted_count: int
    unknown_count: int
    claims: List[Claim]

    @property
    def has_drift(self) -> bool:
        return self.drifted_count > 0

"""Tests for execution context tagging (runs_in) and false positive prevention."""

from pathlib import Path
import pytest

from docdrift.models import Claim, ClaimType, DocSection, Verdict
from docdrift.extractor import RuleBasedExtractor, apply_runs_in_safety_net


def test_quickstart_scaffolding_runs_in_generated_app():
    """Commands inside quickstart creating an app are tagged as runs_in='generated_app'."""
    extractor = RuleBasedExtractor()
    lines = [
        "## Quick Start",
        "",
        "The quickest way to get started with express is to utilize the executable `express(1)` to generate an application as shown below:",
        "",
        "```sh",
        "$ npm install -g express-generator@4",
        "$ express /tmp/foo && cd /tmp/foo",
        "$ npm install",
        "$ npm start",
        "```",
    ]
    section = DocSection(
        doc_path="README.md",
        section_title="Quick Start",
        start_line=1,
        lines=lines
    )

    claims = extractor.extract(section)
    assert len(claims) >= 3

    # Check express /tmp/foo
    express_claims = [c for c in claims if "/tmp/foo" in c.extracted_value or "express" in c.extracted_value]
    assert len(express_claims) > 0
    for c in express_claims:
        assert c.runs_in in ("user_machine", "generated_app")

    # Check npm start in that block
    start_claims = [c for c in claims if "start" in c.extracted_value]
    assert len(start_claims) > 0
    assert start_claims[0].runs_in == "generated_app"


def test_absolute_paths_tagged_as_user_machine():
    """Paths starting with /tmp, ~, C:\\, etc. are tagged as user_machine."""
    claims = [
        Claim(
            doc_path="README.md",
            line_number=5,
            raw_line="Logs saved to `/tmp/app.log`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="/tmp/app.log",
        ),
        Claim(
            doc_path="README.md",
            line_number=6,
            raw_line="Config in `~/.config/app.toml`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="~/.config/app.toml",
        ),
    ]

    safe_claims = apply_runs_in_safety_net(claims)
    assert safe_claims[0].runs_in == "user_machine"
    assert safe_claims[1].runs_in == "user_machine"


def test_external_markdown_links_tagged_as_external():
    """External links like [GOVERNANCE.md](https://...) are tagged as external."""
    claims = [
        Claim(
            doc_path="README.md",
            line_number=12,
            raw_line="See [GOVERNANCE.md](https://github.com/expressjs/discussions/blob/HEAD/docs/GOVERNANCE.md)",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="GOVERNANCE.md",
        )
    ]

    safe_claims = apply_runs_in_safety_net(claims)
    assert safe_claims[0].runs_in == "external"


def test_standard_repo_claims_tagged_as_repo():
    """Standard in-repo commands and files remain tagged as repo."""
    extractor = RuleBasedExtractor()
    lines = [
        "## Development",
        "",
        "Run unit tests:",
        "```sh",
        "npm test",
        "```",
        "Source code located in `src/index.js`.",
    ]
    section = DocSection(
        doc_path="README.md",
        section_title="Development",
        start_line=1,
        lines=lines
    )

    claims = extractor.extract(section)
    for c in claims:
        assert c.runs_in == "repo"

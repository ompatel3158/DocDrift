"""Tests for diff generation, PR metadata, and reporter outputs."""

import os
from pathlib import Path
import pytest

from docdrift.models import Claim, ClaimType, DriftReport, Verdict
from docdrift.reporter import Reporter


def test_pr_description_contains_real_commit_sha_and_repo_url(tmp_path: Path):
    """PR description must contain the real commit SHA and ompatel3158/DocDrift repo URL."""
    doc = tmp_path / "README.md"
    doc.write_text("Old: `npm run dev`\n", encoding="utf-8")

    claim = Claim(
        doc_path=str(doc),
        line_number=1,
        raw_line="Old: `npm run dev`",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run dev",
        verdict=Verdict.DRIFTED,
        evidence="Script 'dev' not found.",
        fixed_line="Old: `npm run start`",
    )

    report = DriftReport(
        repo_path=str(tmp_path),
        files_scanned=[str(doc)],
        total_claims=1,
        verified_count=0,
        drifted_count=1,
        unknown_count=0,
        claims=[claim],
        commit_sha="a1b2c3d4",
    )

    reporter = Reporter()
    pr_desc = reporter.generate_pr_description(report, commit_sha="a1b2c3d4")

    assert "https://github.com/ompatel3158/DocDrift" in pr_desc
    assert "a1b2c3d4" in pr_desc
    assert "`npm run start`" in pr_desc


def test_pr_description_includes_review_section_for_unfixed_claims(tmp_path: Path):
    """PR description cleanly separates actionable fixes from claims requiring maintainer review."""
    doc = tmp_path / "README.md"
    doc.write_text("Unclear claim\n", encoding="utf-8")

    claim = Claim(
        doc_path=str(doc),
        line_number=1,
        raw_line="Unclear claim",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="some_cmd",
        verdict=Verdict.DRIFTED,
        evidence="Command not found.",
        needs_review=True,
        metadata={"rejection_reason": "No safe automated fix found"},
    )

    report = DriftReport(
        repo_path=str(tmp_path),
        files_scanned=[str(doc)],
        total_claims=1,
        verified_count=0,
        drifted_count=1,
        unknown_count=0,
        claims=[claim],
    )

    reporter = Reporter()
    pr_desc = reporter.generate_pr_description(report, commit_sha="main")

    assert "Claims Flagged for Maintainer Review" in pr_desc
    assert "No safe automated fix found" in pr_desc


def test_unified_diff_generates_on_exact_line_number(tmp_path: Path):
    """Unified diff replaces exact line number and formats proper unified diff."""
    doc = tmp_path / "README.md"
    doc.write_text(
        "# Title\n\nRun command:\n```sh\nnpm run dev\n```\n\nFooter.\n",
        encoding="utf-8"
    )

    claim = Claim(
        doc_path=str(doc),
        line_number=5,
        raw_line="npm run dev",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run dev",
        verdict=Verdict.DRIFTED,
        evidence="Script 'dev' not found in package.json.",
        fixed_line="npm run start",
    )

    report = DriftReport(
        repo_path=str(tmp_path),
        files_scanned=[str(doc)],
        total_claims=1,
        verified_count=0,
        drifted_count=1,
        unknown_count=0,
        claims=[claim],
    )

    reporter = Reporter()
    diff_str = reporter.generate_unified_diff(report)

    assert "--- a/README.md" in diff_str
    assert "+++ b/README.md" in diff_str
    assert "-npm run dev" in diff_str
    assert "+npm run start" in diff_str

import difflib
import os
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from docdrift.models import Claim, DriftReport, Verdict, ClaimType


class Reporter:
    """Formats and prints DocDrift results in terminal, diff, and PR formats."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_terminal_report(self, report: DriftReport) -> None:
        """Renders a rich terminal summary report of verification results."""
        self.console.print()

        # Header Badge Panel
        if report.drifted_count == 0:
            badge_color = "green"
            badge_title = "✨ [bold green]DocDrift: All Claims Verified![/bold green]"
            badge_text = (
                f"[green]Scanned [bold]{len(report.files_scanned)}[/bold] document(s) | "
                f"Found [bold]{report.total_claims}[/bold] factual claim(s) | "
                f"All [bold]{report.verified_count}[/bold] claims verified against code.[/green]"
            )
        else:
            badge_color = "red"
            badge_title = "🔍 [bold red]DocDrift: Documentation Drift Detected[/bold red]"
            badge_text = (
                f"Scanned [bold]{len(report.files_scanned)}[/bold] file(s) | "
                f"Total Claims: [bold]{report.total_claims}[/bold] | "
                f"[bold green]{report.verified_count} Verified[/bold green] | "
                f"[bold red]{report.drifted_count} Drifted[/bold red] | "
                f"[dim]{report.unknown_count} Unknown[/dim]"
            )

        self.console.print(Panel(badge_text, title=badge_title, border_style=badge_color))

        if report.drifted_count == 0:
            return

        # Drift Details Table
        table = Table(
            title="Drifted Claims & Proposed Fixes",
            title_style="bold red",
            border_style="dim",
            show_header=True,
            header_style="bold cyan",
            expand=True
        )
        table.add_column("Location", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta", width=14)
        table.add_column("Outdated Claim", style="red")
        table.add_column("Code Verification Proof", style="yellow")
        table.add_column("Gemma 4 Proposed Fix", style="green")

        drifted_claims = [c for c in report.claims if c.verdict == Verdict.DRIFTED]
        for c in drifted_claims:
            rel_path = os.path.relpath(c.doc_path, report.repo_path) if report.repo_path else c.doc_path
            loc_str = f"{rel_path}:{c.line_number}"
            fix_str = c.suggested_replacement or c.fixed_line or "(No fix proposed)"
            table.add_row(
                loc_str,
                c.claim_type.value,
                c.raw_line.strip(),
                c.evidence,
                fix_str.strip()
            )

        self.console.print(table)
        self.console.print()

    def generate_unified_diff(self, report: DriftReport) -> str:
        """Generates unified git diff text for all drifted claims with fixes."""
        diff_outputs: List[str] = []

        # Group claims by document
        claims_by_doc: Dict[str, List[Claim]] = {}
        for c in report.claims:
            if c.verdict == Verdict.DRIFTED and (c.fixed_line or c.suggested_replacement):
                claims_by_doc.setdefault(c.doc_path, []).append(c)

        for doc_path, claims in claims_by_doc.items():
            if not os.path.isfile(doc_path):
                continue

            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    original_lines = f.readlines()
            except Exception:
                continue

            modified_lines = list(original_lines)
            for c in claims:
                idx = c.line_number - 1
                if 0 <= idx < len(modified_lines):
                    replacement = c.fixed_line or c.suggested_replacement or modified_lines[idx]
                    if not replacement.endswith("\n"):
                        replacement += "\n"
                    modified_lines[idx] = replacement

            rel_path = os.path.relpath(doc_path, report.repo_path) if report.repo_path else doc_path
            norm_rel = rel_path.replace("\\", "/")
            diff = difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile=f"a/{norm_rel}",
                tofile=f"b/{norm_rel}",
            )
            diff_text = "".join(diff)
            if diff_text:
                diff_outputs.append(diff_text)

        return "\n".join(diff_outputs)

    def print_diff(self, report: DriftReport) -> None:
        """Prints syntax-highlighted git diff to the console."""
        diff_str = self.generate_unified_diff(report)
        if diff_str:
            self.console.print(Syntax(diff_str, "diff", theme="monokai", line_numbers=False))
        else:
            self.console.print("[dim]No diff to display.[/dim]")

    def apply_fixes(self, report: DriftReport) -> List[str]:
        """Applies proposed fixes in-place to markdown files on disk.

        Returns list of modified document paths.
        """
        modified_files: List[str] = []
        claims_by_doc: Dict[str, List[Claim]] = {}
        for c in report.claims:
            if c.verdict == Verdict.DRIFTED and (c.fixed_line or c.suggested_replacement):
                claims_by_doc.setdefault(c.doc_path, []).append(c)

        for doc_path, claims in claims_by_doc.items():
            if not os.path.isfile(doc_path):
                continue
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                continue

            changed = False
            for c in claims:
                idx = c.line_number - 1
                if 0 <= idx < len(lines):
                    replacement = c.fixed_line or c.suggested_replacement
                    if replacement:
                        if not replacement.endswith("\n"):
                            replacement += "\n"
                        lines[idx] = replacement
                        changed = True

            if changed:
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                modified_files.append(doc_path)

        return modified_files

    def generate_pr_description(self, report: DriftReport, commit_sha: str = "HEAD") -> str:
        """Generates a GitHub PR title and description ready for maintainers."""
        drifted = [c for c in report.claims if c.verdict == Verdict.DRIFTED]
        if not drifted:
            return "No documentation drift detected."

        rows = []
        evidence_rows = []
        for i, c in enumerate(drifted, start=1):
            rel_path = os.path.relpath(c.doc_path, report.repo_path) if report.repo_path else c.doc_path
            fix = c.suggested_replacement or c.fixed_line or ""
            rows.append(f"| `{rel_path}:{c.line_number}` | `{c.raw_line.strip()}` | `{fix.strip()}` |")
            evidence_rows.append(f"- **`{rel_path}:{c.line_number}`** ({c.claim_type.value}): {c.evidence}")

        summary_table = (
            "| File & Line | Outdated Line | Corrected Line |\n"
            "| :--- | :--- | :--- |\n" + "\n".join(rows)
        )
        evidence_list = "\n".join(evidence_rows)

        pr_body = f"""## 📝 Documentation Sync: Fix Outdated Claims

### Why this PR is needed
During onboarding and Hacktoberfest documentation review, **DocDrift** detected **{len(drifted)}** outdated factual claim(s) in the repository documentation that did not match the actual codebase implementation (commands, file paths, or environment variables).

### Summary of Changes
{summary_table}

### Deterministic Code Verification Proof
Every change above was verified against actual repository manifests and source files:
{evidence_list}

---
*Verified against commit `{commit_sha}` using [DocDrift](https://github.com/docdrift/docdrift) powered by Gemma 4 + Deterministic Code Verification.*
"""
        return pr_body

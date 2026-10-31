import difflib
import os
import sys
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
                f"All [bold]{report.verified_count}[/bold] claims verified against code manifests.[/green]\n"
                f"[dim]Precision Guard: 0 false positives on tutorial/scaffolded code.[/dim]"
            )
        else:
            badge_color = "red"
            badge_title = "🔍 [bold red]DocDrift: Documentation Drift Detected[/bold red]"
            badge_text = (
                f"Scanned [bold]{len(report.files_scanned)}[/bold] file(s) | "
                f"Total Claims: [bold]{report.total_claims}[/bold] | "
                f"[bold green]{report.verified_count} Verified[/bold green] | "
                f"[bold red]{report.drifted_count} Drifted[/bold red] "
                f"([green]{report.actionable_drifted_count} validated fixes[/green], "
                f"[yellow]{report.needs_review_count} human review[/yellow]) | "
                f"[dim]{report.unknown_count} Non-Repo / Skipped[/dim]"
            )

        self.console.print(Panel(badge_text, title=badge_title, border_style=badge_color))

        # Drift Details Table
        if report.drifted_count > 0:
            table = Table(
                title="Drifted Claims & Code-Constrained Fixes",
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
            table.add_column("Proposed Fix", style="green")

            drifted_claims = [c for c in report.claims if c.verdict == Verdict.DRIFTED]
            for c in drifted_claims:
                rel_path = os.path.relpath(c.doc_path, report.repo_path) if report.repo_path else c.doc_path
                loc_str = f"{rel_path}:{c.line_number}"
                if c.needs_review or not c.fixed_line:
                    reason = c.metadata.get("rejection_reason", "Needs human review")
                    fix_str = f"[bold yellow]⚠️ Needs human review[/bold yellow]\n[dim]({reason})[/dim]"
                else:
                    fix_str = f"[bold green]{c.fixed_line.strip()}[/bold green]"

                table.add_row(
                    loc_str,
                    c.claim_type.value,
                    c.raw_line.strip(),
                    c.evidence,
                    fix_str
                )

            self.console.print(table)
            self.console.print()

        # Unknown / Non-Repo Claims Table (Explain why claims were skipped)
        unknown_claims = [c for c in report.claims if c.verdict == Verdict.UNKNOWN]
        if unknown_claims:
            un_table = Table(
                title="Non-Repo & Tutorial Contexts Skipped (False Positive Guard)",
                title_style="bold yellow",
                border_style="dim",
                show_header=True,
                header_style="bold yellow",
                expand=True
            )
            un_table.add_column("Location", style="cyan", no_wrap=True)
            un_table.add_column("Type", style="magenta", width=14)
            un_table.add_column("Claim Snippet", style="white")
            un_table.add_column("Context", style="yellow", width=14)
            un_table.add_column("Guard Reason", style="dim")

            for c in unknown_claims:
                rel_path = os.path.relpath(c.doc_path, report.repo_path) if report.repo_path else c.doc_path
                loc_str = f"{rel_path}:{c.line_number}"
                ctx = getattr(c, "runs_in", "unknown")
                un_table.add_row(
                    loc_str,
                    c.claim_type.value,
                    c.extracted_value[:35],
                    ctx,
                    c.evidence or "Skipped non-repo execution context."
                )

            self.console.print(un_table)
            self.console.print()

        # Precision Metrics Line
        self.console.print(
            f"[bold cyan]Precision Guard:[/bold cyan] [green]{report.verified_count} verified[/green] | "
            f"[yellow]{len(unknown_claims)} non-repo contexts guarded[/yellow] | "
            f"[red]{report.drifted_count} drifted[/red] "
            f"({report.actionable_drifted_count} safe fixes, {report.needs_review_count} flagged for maintainer review)\n"
        )

    def generate_unified_diff(self, report: DriftReport) -> str:
        """Generates unified git diff text for all drifted claims with safe fixes."""
        diff_outputs: List[str] = []

        # Group claims by document
        claims_by_doc: Dict[str, List[Claim]] = {}
        for c in report.claims:
            if (
                c.verdict == Verdict.DRIFTED
                and c.fixed_line
                and not c.needs_review
                and c.fixed_line.strip() != c.raw_line.strip()
            ):
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
                    replacement = c.fixed_line
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
        elif report.needs_review_count > 0:
            self.console.print("[yellow]ℹ No automated diff: drifted claim(s) require human review (automated rewrite withheld).[/yellow]")
        else:
            self.console.print("[dim]No diff to display.[/dim]")

    def apply_fixes(self, report: DriftReport) -> List[str]:
        """Applies proposed fixes in-place to markdown files on disk.

        Returns list of modified document paths.
        """
        modified_files: List[str] = []
        claims_by_doc: Dict[str, List[Claim]] = {}
        for c in report.claims:
            if (
                c.verdict == Verdict.DRIFTED
                and c.fixed_line
                and not c.needs_review
                and c.fixed_line.strip() != c.raw_line.strip()
            ):
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
                    replacement = c.fixed_line
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

        actionable = [c for c in drifted if c.fixed_line and not c.needs_review]
        review_needed = [c for c in drifted if c.needs_review or not c.fixed_line]

        rows = []
        evidence_rows = []
        for c in actionable:
            rel_path = os.path.relpath(c.doc_path, report.repo_path) if report.repo_path else c.doc_path
            rows.append(f"| `{rel_path}:{c.line_number}` | `{c.raw_line.strip()}` | `{c.fixed_line.strip()}` |")
            evidence_rows.append(f"- **`{rel_path}:{c.line_number}`** ({c.claim_type.value}): {c.evidence}")

        summary_table = (
            "| File & Line | Outdated Line | Corrected Line |\n"
            "| :--- | :--- | :--- |\n" + "\n".join(rows) if rows else "*No automated changes applied.*"
        )
        evidence_list = "\n".join(evidence_rows) if evidence_rows else "*No automated fixes.*"

        review_section = ""
        if review_needed:
            r_rows = []
            for c in review_needed:
                rel_path = os.path.relpath(c.doc_path, report.repo_path) if report.repo_path else c.doc_path
                reason = c.metadata.get("rejection_reason", "Requires manual author verification")
                r_rows.append(f"| `{rel_path}:{c.line_number}` | `{c.raw_line.strip()}` | {c.evidence} | {reason} |")
            review_section = f"""
### ⚠️ Claims Flagged for Maintainer Review
The following claims differed from repository manifests, but automated rewrites were safely withheld to prevent false positives:
| File & Line | Claim | Evidence | Review Note |
| :--- | :--- | :--- | :--- |
""" + "\n".join(r_rows) + "\n"

        clean_sha = commit_sha[:8] if commit_sha and len(commit_sha) >= 8 else commit_sha
        pr_body = f"""## 📝 Documentation Sync: Fix Outdated Claims

### Why this PR is needed
During onboarding and Hacktoberfest documentation review, **DocDrift** detected **{len(drifted)}** outdated factual claim(s) in the repository documentation that did not match the actual codebase implementation (commands, file paths, or environment variables).

### Summary of Changes
{summary_table}

### Deterministic Code Verification Proof
Every change above was verified against actual repository manifests and source files:
{evidence_list}
{review_section}
---
*Verified against commit `{clean_sha}` using [DocDrift](https://github.com/ompatel3158/DocDrift) powered by Gemma 4 + Deterministic Code Verification.*
"""
        return pr_body

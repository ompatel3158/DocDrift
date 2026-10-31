"""Interactive Split-Diff Reviewer for DocDrift.

Provides a terminal UI using Rich to present a side-by-side Split-Diff
of DRIFTED documentation claims versus proposed Gemma 4 fixes.

Supports:
- Split-Diff side-by-side panels (Left: Outdated line + verification evidence; Right: Proposed fix).
- Interactive user actions: [a] Accept, [e] Edit manually, [s] Skip, [q] Quit.
- In-place immediate file updates preserving exact indentation and newlines.
- Non-interactive mode for automated CI/CD runs and testing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from docdrift.fixer import HybridFixer, RuleBasedFixer
from docdrift.models import Claim, Verdict

logger = logging.getLogger("docdrift.reviewer")


def apply_fix_to_file(doc_path: str, line_number: int, replacement_line: str) -> bool:
    """Update markdown file in-place, preserving indentation and exact newlines.

    Args:
        doc_path: Absolute or relative path to the documentation file.
        line_number: 1-based line number to replace.
        replacement_line: New text content for the line.

    Returns:
        True if updated successfully, False otherwise.
    """
    if not os.path.exists(doc_path):
        logger.error(f"Cannot apply fix: file not found at '{doc_path}'.")
        return False

    try:
        with open(doc_path, "r", encoding="utf-8", newline="") as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Failed to read file '{doc_path}': {e}")
        return False

    idx = line_number - 1
    if not (0 <= idx < len(lines)):
        logger.error(
            f"Line number {line_number} is out of bounds for '{doc_path}' ({len(lines)} total lines)."
        )
        return False

    orig_line = lines[idx]

    # Detect original newline style
    if orig_line.endswith("\r\n"):
        eol = "\r\n"
    elif orig_line.endswith("\n"):
        eol = "\n"
    else:
        eol = ""

    clean_replacement = replacement_line.rstrip("\r\n")

    # Preserve leading indentation if replacement omitted it
    orig_indent = orig_line[: len(orig_line) - len(orig_line.lstrip())]
    if orig_indent and not clean_replacement.startswith((" ", "\t")):
        clean_replacement = orig_indent + clean_replacement

    lines[idx] = clean_replacement + eol

    try:
        with open(doc_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        return True
    except Exception as e:
        logger.error(f"Failed to write updated file '{doc_path}': {e}")
        return False


class InteractiveReviewer:
    """Rich-based Interactive Split-Diff Reviewer for DRIFTED claims."""

    def __init__(
        self,
        console: Optional[Console] = None,
        non_interactive: bool = False,
        auto_action: Optional[Union[str, Callable[[Claim], str]]] = None,
        input_func: Optional[Callable[[str], str]] = None,
        fixer: Optional[Any] = None,
    ) -> None:
        """Initialize InteractiveReviewer.

        Args:
            console: Optional Rich Console.
            non_interactive: If True, executes without blocking for user input.
            auto_action: Action string ("accept", "skip") or callback for non-interactive mode.
            input_func: Custom input function for mocking or piping interactive responses.
            fixer: Optional fixer to compute fixed_line if missing.
        """
        self.console = console or Console()
        self.non_interactive = non_interactive
        self.auto_action = auto_action
        self.input_func = input_func
        self.fixer = fixer or RuleBasedFixer()

    def create_split_diff_view(
        self,
        claim: Claim,
        index: int = 1,
        total: int = 1,
    ) -> Table:
        """Render side-by-side Split-Diff panels comparing outdated line with proposed fix.

        Args:
            claim: The drifted Claim.
            index: Current claim index.
            total: Total number of drifted claims.

        Returns:
            Rich Table containing the side-by-side diff panels.
        """
        # Ensure fixed line is available
        proposed_line = claim.fixed_line or claim.suggested_replacement or claim.raw_line

        # Left Panel: Outdated line + verification evidence
        left_body = Table.grid(padding=(0, 1))
        left_body.add_column(style="bold")
        left_body.add_column()

        left_body.add_row("Document:", Text(f"{claim.doc_path}:{claim.line_number}", style="yellow"))
        left_body.add_row("Claim Type:", Text(str(claim.claim_type.value), style="cyan"))
        left_body.add_row("Extracted:", Text(str(claim.extracted_value), style="red bold"))
        left_body.add_row("Outdated Line:", Syntax(claim.raw_line.strip(), "markdown", background_color="default"))
        left_body.add_row(
            "Proof Evidence:",
            Text(claim.evidence or "Code check proved claim is outdated/false", style="red"),
        )

        left_panel = Panel(
            left_body,
            title=f"[bold red]❌ Drifted (Line {claim.line_number})[/bold red]",
            border_style="red",
            subtitle=f"[dim]{claim.doc_path}[/dim]",
        )

        # Right Panel: Proposed fix
        right_body = Table.grid(padding=(0, 1))
        right_body.add_column(style="bold")
        right_body.add_column()

        right_body.add_row("Fix Engine:", Text("Gemma 4 Surgical Rewriter", style="green bold"))
        right_body.add_row(
            "Proposed Value:",
            Text(str(claim.suggested_replacement or "Contextual Rewrite"), style="green bold"),
        )
        right_body.add_row("Proposed Line:", Syntax(proposed_line.strip(), "markdown", background_color="default"))

        right_panel = Panel(
            right_body,
            title="[bold green]✨ Proposed Fix (Gemma 4)[/bold green]",
            border_style="green",
            subtitle="[dim]Context-Preserving[/dim]",
        )

        # Split-Diff Grid Layout (50/50 ratio)
        diff_grid = Table.grid(expand=True)
        diff_grid.add_column(ratio=1)
        diff_grid.add_column(ratio=1)
        diff_grid.add_row(left_panel, right_panel)

        container = Table.grid(expand=True)
        container.add_column()
        container.add_row(
            Panel(
                diff_grid,
                title=f"[bold]Drift Review ({index}/{total})[/bold]",
                border_style="blue",
            )
        )

        return container

    def prompt_action(self, claim: Claim) -> str:
        """Prompt user for an action or resolve via non-interactive policy.

        Returns:
            One of: 'a' (accept), 'e' (edit), 's' (skip), 'q' (quit).
        """
        if self.non_interactive:
            if callable(self.auto_action):
                return self.auto_action(claim).lower().strip()
            if isinstance(self.auto_action, str):
                act = self.auto_action.lower().strip()
                if act in {"accept", "a"}:
                    return "a"
                if act in {"skip", "s"}:
                    return "s"
                if act in {"quit", "q"}:
                    return "q"
            return "s"

        prompt_text = "[bold cyan]Action: [a] Accept | [e] Edit manually | [s] Skip | [q] Quit:[/bold cyan] "
        while True:
            if self.input_func:
                raw_choice = self.input_func(prompt_text)
            else:
                raw_choice = self.console.input(prompt_text)

            choice = raw_choice.strip().lower()
            if choice in {"a", "accept"}:
                return "a"
            if choice in {"e", "edit"}:
                return "e"
            if choice in {"s", "skip", ""}:
                return "s"
            if choice in {"q", "quit"}:
                return "q"

            self.console.print(
                "[yellow]Invalid selection. Please choose 'a', 'e', 's', or 'q'.[/yellow]"
            )

    def review_claim(self, claim: Claim, index: int = 1, total: int = 1) -> str:
        """Present split-diff review for a single claim and handle the action.

        Args:
            claim: The drifted Claim.
            index: Current claim index.
            total: Total drifted claims count.

        Returns:
            Result status: 'accepted', 'edited', 'skipped', or 'quit'.
        """
        # Ensure fix exists
        if not claim.fixed_line and hasattr(self.fixer, "fix_claim"):
            try:
                self.fixer.fix_claim(claim)
            except Exception as e:
                logger.warning(f"Failed to generate fix line for review: {e}")

        # Render split-diff view
        split_diff = self.create_split_diff_view(claim, index=index, total=total)
        self.console.print(split_diff)

        action = self.prompt_action(claim)

        if action == "a":
            target_line = claim.fixed_line or claim.suggested_replacement or claim.raw_line
            success = apply_fix_to_file(claim.doc_path, claim.line_number, target_line)
            if success:
                self.console.print(
                    f"[bold green]✓ Accepted & Applied fix to {claim.doc_path}:{claim.line_number}[/bold green]\n"
                )
                return "accepted"
            else:
                self.console.print(
                    f"[bold red]✗ Failed to update file at {claim.doc_path}:{claim.line_number}[/bold red]\n"
                )
                return "skipped"

        elif action == "e":
            default_val = claim.fixed_line or claim.raw_line
            if self.non_interactive:
                custom_line = default_val
            elif self.input_func:
                custom_line = self.input_func(f"[cyan]Enter replacement line (default: {default_val.strip()}): [/cyan]")
            else:
                custom_line = self.console.input(f"[cyan]Enter replacement line (default: {default_val.strip()}): [/cyan]")

            custom_line = custom_line.strip() or default_val
            claim.fixed_line = custom_line
            success = apply_fix_to_file(claim.doc_path, claim.line_number, custom_line)
            if success:
                self.console.print(
                    f"[bold green]✓ Applied manual edit to {claim.doc_path}:{claim.line_number}[/bold green]\n"
                )
                return "edited"
            else:
                self.console.print(
                    f"[bold red]✗ Failed to update file at {claim.doc_path}:{claim.line_number}[/bold red]\n"
                )
                return "skipped"

        elif action == "s":
            self.console.print(
                f"[bold yellow]↷ Skipped {claim.doc_path}:{claim.line_number}[/bold yellow]\n"
            )
            return "skipped"

        elif action == "q":
            self.console.print("[bold red]Review session aborted by user.[/bold red]\n")
            return "quit"

        return "skipped"

    def review_claims(self, claims: List[Claim]) -> Dict[str, int]:
        """Review all DRIFTED claims in an interactive Split-Diff session.

        Args:
            claims: List of Claim instances.

        Returns:
            Dictionary with counts of accepted, edited, skipped, and total claims.
        """
        drifted = [c for c in claims if c.verdict == Verdict.DRIFTED]
        stats = {"accepted": 0, "edited": 0, "skipped": 0, "total": len(drifted)}

        if not drifted:
            self.console.print("[green]No DRIFTED claims found! Documentation is up to date.[/green]")
            return stats

        self.console.print(
            f"\n[bold blue]Starting Interactive Split-Diff Review ({len(drifted)} drifted claims)...[/bold blue]\n"
        )

        for i, claim in enumerate(drifted, start=1):
            decision = self.review_claim(claim, index=i, total=len(drifted))
            if decision == "quit":
                break
            elif decision == "accepted":
                stats["accepted"] += 1
            elif decision == "edited":
                stats["edited"] += 1
            elif decision == "skipped":
                stats["skipped"] += 1

        # Summary Table
        summary_table = Table(title="Drift Review Summary", border_style="blue")
        summary_table.add_column("Status", style="bold")
        summary_table.add_column("Count", justify="right")

        summary_table.add_row("Total Drifted", str(stats["total"]))
        summary_table.add_row("[green]Accepted[/green]", f"[green]{stats['accepted']}[/green]")
        summary_table.add_row("[cyan]Manually Edited[/cyan]", f"[cyan]{stats['edited']}[/cyan]")
        summary_table.add_row("[yellow]Skipped[/yellow]", f"[yellow]{stats['skipped']}[/yellow]")

        self.console.print(summary_table)
        return stats

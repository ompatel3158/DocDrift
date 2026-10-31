import os
import time
from datetime import datetime
from typing import Callable, Optional, List
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table

from docdrift.git_helper import GitHelper
from docdrift.models import DriftReport


class WatchLoop:
    """Manages the continuous watch loop, file monitoring, and periodic Git pulling."""

    def __init__(
        self,
        repo_path: str,
        verify_callback: Callable[[], DriftReport],
        auto_fix: bool = False,
        pull_interval_sec: Optional[int] = None,
        poll_interval_sec: float = 2.0,
        console: Optional[Console] = None
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.verify_callback = verify_callback
        self.auto_fix = auto_fix
        self.pull_interval_sec = pull_interval_sec
        self.poll_interval_sec = poll_interval_sec
        self.console = console or Console()
        self._last_pull_time = 0.0
        self._iteration = 0
        self._file_mtimes = {}

    def _get_tracked_files_mtime(self) -> float:
        """Returns the maximum mtime across markdown files and manifests in the repo."""
        max_mtime = 0.0
        for root, dirs, files in os.walk(self.repo_path):
            # Ignore hidden and build dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "dist", "build", "__pycache__")]
            for f in files:
                if f.lower().endswith((".md", ".json", ".toml", ".yml", ".yaml")) or f in ("Makefile", "Dockerfile"):
                    full_p = os.path.join(root, f)
                    try:
                        mtime = os.path.getmtime(full_p)
                        if mtime > max_mtime:
                            max_mtime = mtime
                    except OSError:
                        pass
        return max_mtime

    def _render_dashboard(self, report: Optional[DriftReport], status_msg: str, git_msg: str) -> Panel:
        """Renders a live rich panel for the watch loop."""
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")

        now_str = datetime.now().strftime("%H:%M:%S")
        grid.add_row(
            f"[bold cyan]DocDrift Watcher[/bold cyan] (Loop #{self._iteration})",
            f"[dim]{now_str}[/dim]"
        )

        stats_str = ""
        if report:
            if report.drifted_count == 0:
                stats_str = f"[bold green]✓ 0 Drifted[/bold green] | [dim]{report.verified_count} verified[/dim]"
            else:
                stats_str = f"[bold red]✗ {report.drifted_count} DRIFTED[/bold red] | [green]{report.verified_count} verified[/green]"

        grid.add_row(f"[bold]Target:[/bold] [yellow]{os.path.basename(self.repo_path)}[/yellow]", stats_str)
        grid.add_row(f"[bold]Status:[/bold] {status_msg}", f"[dim]Pull: {git_msg}[/dim]")

        border_style = "green" if (report and report.drifted_count == 0) else "yellow"
        if report and report.drifted_count > 0:
            border_style = "red"

        return Panel(grid, border_style=border_style, title="[bold]DocDrift Continuous Daemon[/bold]")

    def run_loop(self, max_iterations: Optional[int] = None) -> None:
        """Executes the continuous monitoring loop."""
        self.console.print(f"[bold green]Starting DocDrift continuous watch on:[/bold green] {self.repo_path}")
        if self.pull_interval_sec:
            self.console.print(f"[dim]Git remote pulling active (every {self.pull_interval_sec}s)[/dim]")

        last_known_mtime = self._get_tracked_files_mtime()
        current_report = None
        git_msg = "Inactive" if not self.pull_interval_sec else "Pending initial pull"

        # Initial verification run
        try:
            current_report = self.verify_callback()
        except Exception as e:
            self.console.print(f"[red]Initial scan error: {e}[/red]")

        with Live(
            self._render_dashboard(current_report, "Monitoring repository files...", git_msg),
            console=self.console,
            refresh_per_second=2
        ) as live:
            while True:
                self._iteration += 1
                if max_iterations and self._iteration >= max_iterations:
                    break

                now = time.time()
                status_msg = "Listening for file modifications..."

                # Check periodic Git pull if configured
                if self.pull_interval_sec and (now - self._last_pull_time >= self.pull_interval_sec):
                    self._last_pull_time = now
                    has_updates, pull_res = GitHelper.pull_updates(self.repo_path)
                    if has_updates:
                        git_msg = f"New commits pulled ({now_str})"
                        status_msg = "[cyan]New git updates detected, re-verifying...[/cyan]"
                        live.update(self._render_dashboard(current_report, status_msg, git_msg))
                        current_report = self.verify_callback()
                    else:
                        git_msg = "Up-to-date"

                # Check local filesystem modifications
                current_mtime = self._get_tracked_files_mtime()
                if current_mtime > last_known_mtime:
                    last_known_mtime = current_mtime
                    status_msg = "[yellow]Local documentation/code change detected! Re-verifying...[/yellow]"
                    live.update(self._render_dashboard(current_report, status_msg, git_msg))
                    current_report = self.verify_callback()
                    status_msg = "[green]Verification up to date.[/green]"

                live.update(self._render_dashboard(current_report, status_msg, git_msg))
                time.sleep(self.poll_interval_sec)

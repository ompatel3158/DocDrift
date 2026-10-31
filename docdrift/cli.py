import json
import os
import sys
from typing import List, Optional

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from docdrift.models import Claim, ClaimType, Verdict, DriftReport
from docdrift.scanner import DocScanner
from docdrift.git_helper import GitHelper
from docdrift.reporter import Reporter
from docdrift.loop import WatchLoop


app = typer.Typer(
    name="docdrift",
    help="AI + deterministic code verifier that catches outdated README and docs claims and fixes them.",
    add_completion=False,
)
console = Console()


def run_pipeline(
    target: str,
    model: str = "gemma-4-26b-it",
    offline: bool = False,
    probe_commands: bool = True,
    prefer_ultsql: bool = True,
) -> DriftReport:
    """Core verification pipeline execution."""
    # 1. Resolve repository path (handling remote Git URLs if provided)
    is_temp_clone = False
    clean_target = target.strip()
    if GitHelper.is_git_url(clean_target):
        console.print(f"[cyan]Cloning remote repository:[/cyan] {clean_target}")
        repo_path, is_temp_clone = GitHelper.clone_repo(clean_target)
    else:
        repo_path = os.path.abspath(clean_target)

    if not os.path.exists(repo_path):
        console.print(f"[bold red]Error:[/bold red] Target path '{repo_path}' does not exist.")
        raise typer.Exit(code=1)

    try:
        # 2. Initialize Memory Backend (UltSQL with local fallback)
        from docdrift.memory import get_memory_backend
        memory = get_memory_backend(repo_path, prefer_ultsql=prefer_ultsql)

        # 3. Discover documentation files
        scanner = DocScanner(repo_path)
        doc_files = scanner.find_doc_files()

        if not doc_files:
            console.print("[yellow]No documentation files (README or docs/*.md) found.[/yellow]")
            return DriftReport(
                repo_path=repo_path,
                files_scanned=[],
                total_claims=0,
                verified_count=0,
                drifted_count=0,
                unknown_count=0,
                claims=[]
            )

        # 4. Initialize AI Client & Extractor
        from docdrift.ai_client import GemmaAIClient
        from docdrift.extractor import HybridExtractor
        ai_client = GemmaAIClient(model=model, offline=offline)
        extractor = HybridExtractor(ai_client=ai_client, memory=memory)

        # 5. Extract claims across documents and sections
        all_claims: List[Claim] = []
        for doc_file in doc_files:
            sections = scanner.parse_sections(doc_file)
            for section in sections:
                # Check memory cache: if section hash is already cached and unchanged, reuse claims
                claims = extractor.extract_claims(section)
                all_claims.extend(claims)

        # 6. Initialize Deterministic Verifiers
        from docdrift.verifiers.file_verifier import FileVerifier
        from docdrift.verifiers.command_verifier import MultiManifestVerifier
        from docdrift.verifiers.env_verifier import EnvVerifier
        from docdrift.verifiers.sandbox_prober import SandboxCommandProber

        file_verifier = FileVerifier()
        cmd_verifier = MultiManifestVerifier()
        env_verifier = EnvVerifier(memory=memory)
        sandbox_prober = SandboxCommandProber() if probe_commands else None

        verified_count = 0
        drifted_count = 0
        unknown_count = 0

        # 7. Deterministic Verification Loop (Pure Python Code - No AI Hallucination)
        for claim in all_claims:
            v_res = None
            if file_verifier.can_verify(claim):
                v_res = file_verifier.verify(claim, repo_path)
            elif cmd_verifier.can_verify(claim):
                v_res = cmd_verifier.verify(claim, repo_path)
                # If command exists in manifest, run optional safe sandbox probe
                if v_res.verdict == Verdict.VERIFIED and sandbox_prober:
                    probe_res = sandbox_prober.probe_command(claim.extracted_value, repo_path)
                    if probe_res.evidence:
                        v_res.evidence += f" | {probe_res.evidence}"
            elif env_verifier.can_verify(claim):
                v_res = env_verifier.verify(claim, repo_path)

            if v_res:
                claim.verdict = v_res.verdict
                claim.evidence = v_res.evidence
                claim.suggested_replacement = v_res.suggested_value
            else:
                claim.verdict = Verdict.UNKNOWN
                claim.evidence = "No specialized verifier matched this claim type."

            # Update counters
            if claim.verdict == Verdict.VERIFIED:
                verified_count += 1
            elif claim.verdict == Verdict.DRIFTED:
                drifted_count += 1
            else:
                unknown_count += 1

            # Save claim to memory store
            memory.store_claim(claim)

        # 8. Fix Generation for Drifted Claims (Gemma 4 surgical rewrite)
        from docdrift.fixer import GemmaFixer
        fixer = GemmaFixer(ai_client=ai_client)
        for claim in all_claims:
            if claim.verdict == Verdict.DRIFTED:
                claim.fixed_line = fixer.generate_fix(claim)

        report = DriftReport(
            repo_path=repo_path,
            files_scanned=doc_files,
            total_claims=len(all_claims),
            verified_count=verified_count,
            drifted_count=drifted_count,
            unknown_count=unknown_count,
            claims=all_claims
        )

        return report

    finally:
        if is_temp_clone:
            GitHelper.cleanup_temp_repo(repo_path)


@app.command()
def check(
    target: str = typer.Argument(".", help="Path to local repository or remote GitHub URL"),
    fix: bool = typer.Option(False, "--fix", "-f", help="Automatically apply suggested fixes in-place"),
    diff: bool = typer.Option(False, "--diff", "-d", help="Display unified git diff of proposed fixes"),
    pr: bool = typer.Option(False, "--pr", help="Generate a ready-to-merge GitHub PR description"),
    model: str = typer.Option("gemma-4-26b-it", "--model", "-m", help="Gemma 4 model: gemma-4-26b-it or gemma-4-31b-it"),
    offline: bool = typer.Option(False, "--offline", help="Run in offline heuristic mode (skips Gemini API calls)"),
    probe: bool = typer.Option(True, "--probe/--no-probe", help="Safely probe command execution and toolchains on PATH"),
    ultsql: bool = typer.Option(True, "--ultsql/--no-ultsql", help="Use UltSQL converged memory engine with fallback"),
    json_output: Optional[str] = typer.Option(None, "--json", "-j", help="Path to write JSON verification report"),
):
    """Scan and verify documentation claims against the repository codebase."""
    console.print(f"[bold cyan]DocDrift v0.1.0[/bold cyan] checking: [yellow]{target}[/yellow]")
    report = run_pipeline(
        target=target,
        model=model,
        offline=offline,
        probe_commands=probe,
        prefer_ultsql=ultsql
    )

    reporter = Reporter(console=console)
    reporter.print_terminal_report(report)

    if diff or (fix and report.has_drift):
        console.print("[bold cyan]Git Unified Diff:[/bold cyan]")
        reporter.print_diff(report)

    if fix and report.has_drift:
        modified = reporter.apply_fixes(report)
        console.print(f"[bold green]Successfully updated {len(modified)} file(s) in place.[/bold green]")

    if pr and report.has_drift:
        commit_sha = GitHelper.get_commit_hash(report.repo_path)
        pr_desc = reporter.generate_pr_description(report, commit_sha=commit_sha)
        console.print(Panel(pr_desc, title="[bold green]Ready-to-Use GitHub PR Description[/bold green]", border_style="green"))

    if json_output:
        with open(json_output, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        console.print(f"[dim]Report saved to: {json_output}[/dim]")

    if report.has_drift and not fix:
        raise typer.Exit(code=1)


@app.command()
def review(
    target: str = typer.Argument(".", help="Path to local repository or remote GitHub URL"),
    model: str = typer.Option("gemma-4-26b-it", "--model", "-m", help="Gemma 4 model"),
    offline: bool = typer.Option(False, "--offline", help="Run offline without API key"),
):
    """Interactively review and accept/edit/skip drifted claims one by one."""
    from docdrift.reviewer import InteractiveReviewer

    report = run_pipeline(target=target, model=model, offline=offline)
    if not report.has_drift:
        console.print("[bold green]✨ No documentation drift found to review![/bold green]")
        return

    reviewer = InteractiveReviewer(console=console)
    reviewer.review_report(report)


@app.command()
def watch(
    target: str = typer.Argument(".", help="Path to local repository"),
    pull: Optional[int] = typer.Option(None, "--pull", "-p", help="Interval in seconds to poll & pull git remote updates"),
    fix: bool = typer.Option(False, "--fix", "-f", help="Automatically auto-fix drifted lines on detection"),
    model: str = typer.Option("gemma-4-26b-it", "--model", "-m", help="Gemma 4 model"),
    offline: bool = typer.Option(False, "--offline", help="Run offline without API key"),
):
    """Continuously monitor documentation and manifests in a daemon loop."""
    def callback() -> DriftReport:
        rep = run_pipeline(target=target, model=model, offline=offline)
        if fix and rep.has_drift:
            Reporter(console=console).apply_fixes(rep)
        return rep

    watcher = WatchLoop(
        repo_path=target,
        verify_callback=callback,
        auto_fix=fix,
        pull_interval_sec=pull,
        console=console
    )
    watcher.run_loop()


@app.command()
def memory(
    target: str = typer.Argument(".", help="Path to local repository"),
    clear: bool = typer.Option(False, "--clear", help="Clear the persistent memory cache"),
):
    """Inspect or reset DocDrift's persistent UltSQL/local memory cache."""
    from docdrift.memory import get_memory_backend
    repo_path = os.path.abspath(target)
    mem = get_memory_backend(repo_path)

    if clear:
        mem.clear()
        console.print(f"[bold green]Cleared DocDrift memory cache for:[/bold green] {repo_path}")
        return

    stats = mem.get_stats()
    table = Table(title="DocDrift Memory Engine Stats", border_style="cyan")
    table.add_column("Property", style="bold yellow")
    table.add_column("Value", style="green")

    for k, v in stats.items():
        table.add_row(str(k), str(v))

    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()

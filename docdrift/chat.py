"""Interactive CLI Chat Session for DocDrift.

Enables conversational Q&A, on-demand codebase verification, diff inspection,
in-place documentation fixes, and live terminal interaction powered by Gemma 4.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from docdrift.ai_client import GemmaAIClient, load_env_file
from docdrift.git_helper import GitHelper
from docdrift.models import Claim, ClaimType, DriftReport, Verdict, VerificationResult
from docdrift.reporter import Reporter
from docdrift.scanner import DocScanner
from docdrift.verifiers.command_verifier import MultiManifestVerifier
from docdrift.verifiers.dependency_verifier import DependencyVerifier
from docdrift.verifiers.env_verifier import EnvVerifier
from docdrift.verifiers.file_verifier import FileVerifier
from docdrift.verifiers.sandbox_prober import SandboxCommandProber


class InteractiveChat:
    """Manages an interactive multi-turn CLI chat session with DocDrift."""

    def __init__(
        self,
        repo_path: str = ".",
        model: str = "gemma-4-26b-it",
        offline: bool = False,
        probe_commands: bool = True,
        prefer_ultsql: bool = True,
        console: Optional[Console] = None,
    ):
        self.raw_target = repo_path
        self.is_temp_clone = False
        self.console = console or Console()
        self.model = model
        self.offline = offline
        self.probe_commands = probe_commands
        self.prefer_ultsql = prefer_ultsql

        # 1. Resolve repository path
        clean_target = repo_path.strip()
        self.target = clean_target
        if GitHelper.is_git_url(clean_target):
            with self.console.status(f"[bold cyan]Cloning remote repository:[/bold cyan] {clean_target} ...", spinner="dots"):
                self.repo_path, self.is_temp_clone = GitHelper.clone_repo(clean_target)
            self.console.print(f"  [green]✔[/green] Cloned shallow repository into temporary sandbox")
            url_repo = clean_target.rstrip("/").split("/")[-1].replace(".git", "")
            self.repo_name = url_repo if url_repo else os.path.basename(self.repo_path)
        else:
            self.repo_path = os.path.abspath(clean_target)
            self.repo_name = os.path.basename(self.repo_path)

        if not os.path.exists(self.repo_path):
            raise FileNotFoundError(f"Target repository path '{self.repo_path}' does not exist.")

        # Load environment variables
        load_env_file([
            os.path.join(self.repo_path, ".env"),
            os.path.join(os.getcwd(), ".env"),
        ])
        if os.environ.get("DOCDRIFT_MODEL") and model == "gemma-4-26b-it":
            self.model = os.environ.get("DOCDRIFT_MODEL")

        # 2. Initialize Memory & AI Client
        from docdrift.memory import get_memory_backend
        self.memory = get_memory_backend(self.repo_path, prefer_ultsql=prefer_ultsql)
        self.ai_client = GemmaAIClient(model=self.model, offline=offline)

        # 3. Initialize Verifiers
        self.file_verifier = FileVerifier()
        self.cmd_verifier = MultiManifestVerifier()
        self.env_verifier = EnvVerifier(memory=self.memory)
        self.dep_verifier = DependencyVerifier()
        self.sandbox_prober = SandboxCommandProber() if probe_commands else None

        # 4. State
        self.scanner = DocScanner(self.repo_path)
        self.doc_files: List[str] = self.scanner.find_doc_files()
        self.commit_sha = GitHelper.get_commit_hash(self.repo_path)
        self.current_report: Optional[DriftReport] = None
        self.history: List[Dict[str, str]] = []

        # 5. Setup Gemma 4 multi-turn chat if available
        self._genai_chat = None
        if not self.offline and not self.ai_client.is_offline and self.ai_client._client is not None:
            try:
                from google.genai import types
                sys_inst = self._build_system_instruction()
                config = types.GenerateContentConfig(
                    system_instruction=sys_inst,
                    temperature=0.3,
                )
                self._genai_chat = self.ai_client._client.chats.create(
                    model=self.ai_client.model,
                    config=config,
                )
            except Exception as e:
                self.console.print(f"[dim yellow]Notice: Native Gemma chat session initialization fell back to prompt mode: {e}[/dim yellow]")

    # --------------------------------------------------------------------------
    # Banner & Context
    # --------------------------------------------------------------------------

    def _render_welcome_banner(self) -> None:
        """Renders rich greeting panel with repo status and shortcuts."""
        repo_name = getattr(self, "repo_name", os.path.basename(self.repo_path))
        doc_count = len(self.doc_files)
        model_label = f"Offline Mode" if self.offline else f"Gemma 4 ({self.model})"
        engine_label = "UltSQL Converged" if self.prefer_ultsql else "Local SQLite"

        banner_text = (
            f"[bold cyan]Repository:[/bold cyan] [yellow]{repo_name}[/yellow] ([dim]{self.commit_sha[:7] if self.commit_sha else 'local'}[/dim])\n"
            f"[bold cyan]Documentation Files:[/bold cyan] {doc_count} detected | [bold cyan]AI Engine:[/bold cyan] {model_label}\n"
            f"[bold cyan]Memory Engine:[/bold cyan] {engine_label} | [bold cyan]Code Verification:[/bold cyan] [green]Active[/green]\n\n"
            f"[dim]Ask questions in plain English or use slash commands:[/dim]\n"
            f"  [bold green]/check[/bold green]      Run full verification scan on repository documentation\n"
            f"  [bold green]/test[/bold green]       Verify code health, syntax, entrypoints & dependencies\n"
            f"  [bold green]/drift[/bold green]      Show current drifted claims and proof\n"
            f"  [bold green]/fix[/bold green]        Apply safe code-constrained fixes in-place\n"
            f"  [bold green]/diff[/bold green]       View unified git diff of proposed fixes\n"
            f"  [bold green]/pr[/bold green]         Generate ready-to-merge GitHub PR description\n"
            f"  [bold green]/verify[/bold green]     Test any claim on the fly ([dim]e.g. /verify npm test[/dim])\n"
            f"  [bold green]/files[/bold green]      List scanned docs and discovered manifests\n"
            f"  [bold green]/help[/bold green]       Show all commands | [bold green]/exit[/bold green] Quit chat session"
        )

        self.console.print(
            Panel(
                banner_text,
                title="[bold cyan]💬 DocDrift Interactive CLI Chat[/bold cyan]",
                border_style="cyan",
            )
        )

    def _build_system_instruction(self) -> str:
        """Generates system instruction giving Gemma 4 codebase ground truth."""
        repo_name = getattr(self, "repo_name", os.path.basename(self.repo_path))
        doc_names = [os.path.relpath(p, self.repo_path) for p in self.doc_files]

        # Read the primary README excerpt (first 100 lines / ~3KB) if available
        readme_summary = ""
        for doc in self.doc_files:
            if os.path.basename(doc).lower() in ("readme.md", "readme"):
                try:
                    with open(doc, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [f.readline() for _ in range(100)]
                        readme_summary = "".join(lines).strip()
                        if readme_summary:
                            break
                except Exception:
                    pass

        # Collect manifest scripts and targets
        scripts_info = []
        node_scripts = self.cmd_verifier._get_node_scripts(Path(self.repo_path))
        if node_scripts:
            scripts_info.append(f"package.json scripts: {list(node_scripts.keys())}")
        py_scripts = self.cmd_verifier._get_python_scripts(Path(self.repo_path))
        if py_scripts:
            scripts_info.append(f"Python pyproject/setup scripts: {list(py_scripts.keys())}")
        make_targets = self.cmd_verifier._get_make_targets(Path(self.repo_path))
        if make_targets:
            scripts_info.append(f"Makefile targets: {list(make_targets.keys())}")

        readme_block = f"\n\nPrimary README Overview:\n\"\"\"\n{readme_summary}\n\"\"\"\n" if readme_summary else ""

        return (
            f"You are DocDrift, an intelligent documentation synchronization and codebase verifier assistant. "
            f"You are assisting a developer in repository '{repo_name}' (commit {self.commit_sha}).\n"
            f"Repository documentation files: {', '.join(doc_names) if doc_names else 'None'}.\n"
            f"Manifest information: {'; '.join(scripts_info) if scripts_info else 'None'}."
            f"{readme_block}\n"
            f"Guidelines:\n"
            f"1. Be concise, direct, helpful, and technically accurate.\n"
            f"2. Every claim in docs must align with code manifests and actual source code.\n"
            f"3. When answering about what the project is about, summarize the purpose, key features, architecture, and tech stack using the README and codebase ground truth.\n"
            f"4. When answering about drift or outdated instructions, cite the exact file and line number.\n"
            f"5. You can instruct the user to run slash commands such as /check, /test, /fix, /diff, or /pr to take action.\n"
            f"6. Do not hallucinate files or scripts that do not exist in the repository."
        )

    # --------------------------------------------------------------------------
    # Main Chat Loop
    # --------------------------------------------------------------------------

    def start(self) -> None:
        """Starts the interactive CLI chat session."""
        self._render_welcome_banner()

        while True:
            try:
                user_input = self.console.input("\n[bold cyan]docdrift[/bold cyan] [bold green]❯[/bold green] ").strip()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]Exiting DocDrift chat session. Goodbye![/dim]")
                break

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ("/exit", "/quit", "exit", "quit", ":q"):
                self.console.print("[dim]Exiting DocDrift chat session. Goodbye![/dim]")
                break

            # Handle slash commands or natural language query
            try:
                if user_input.startswith("/"):
                    self._handle_slash_command(user_input)
                else:
                    self._handle_natural_query(user_input)
            except Exception as e:
                self.console.print(f"[bold red]Error executing command:[/bold red] {e}")

        # Cleanup memory backend and remote temporary clone
        if hasattr(self, "memory") and self.memory:
            try:
                self.memory.close()
            except Exception:
                pass
        if self.is_temp_clone:
            with self.console.status("[bold cyan]Cleaning up temporary clone sandbox...[/bold cyan]", spinner="dots"):
                GitHelper.cleanup_temp_repo(self.repo_path)

    # --------------------------------------------------------------------------
    # Slash Command Handlers
    # --------------------------------------------------------------------------

    def _handle_slash_command(self, cmd_line: str) -> None:
        """Dispatches slash command entered by user."""
        parts = cmd_line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/help", "/h", "/?"):
            self._cmd_help()
        elif cmd in ("/check", "/scan"):
            self._cmd_check(arg)
        elif cmd in ("/test", "/health"):
            self._cmd_test(arg)
        elif cmd in ("/drift", "/status"):
            self._cmd_drift()
        elif cmd == "/fix":
            self._cmd_fix()
        elif cmd == "/diff":
            self._cmd_diff()
        elif cmd == "/pr":
            self._cmd_pr()
        elif cmd == "/verify":
            self._cmd_verify(arg)
        elif cmd == "/files":
            self._cmd_files()
        elif cmd == "/manifests":
            self._cmd_manifests()
        elif cmd == "/clear":
            self.history.clear()
            os.system("cls" if os.name == "nt" else "clear")
            self._render_welcome_banner()
        else:
            self.console.print(f"[yellow]Unknown command '{cmd}'. Type [bold]/help[/bold] to see available commands.[/yellow]")

    def _cmd_help(self) -> None:
        """Displays interactive command reference."""
        table = Table(title="DocDrift Chat Commands", border_style="cyan", show_header=True)
        table.add_column("Command", style="bold green", width=18)
        table.add_column("Description", style="white")
        table.add_column("Example", style="dim cyan")

        table.add_row("/check [file]", "Run verification scan across repository or single file", "/check backend/README.md")
        table.add_row("/test", "Verify codebase health: syntax, entrypoints, dependencies & tests", "/test")
        table.add_row("/drift", "Display currently detected documentation drift and proofs", "/drift")
        table.add_row("/fix", "Apply validated code-constrained fixes in-place", "/fix")
        table.add_row("/diff", "Show unified git diff of proposed line replacements", "/diff")
        table.add_row("/pr", "Generate ready-to-merge GitHub PR description", "/pr")
        table.add_row("/verify <claim>", "Verify any command, file path, port, or env variable", "/verify npm run build")
        table.add_row("/files", "List scanned documentation files and repository manifests", "/files")
        table.add_row("/manifests", "Inspect discovered scripts and dependencies", "/manifests")
        table.add_row("/clear", "Clear chat history and reset screen", "/clear")
        table.add_row("/exit", "Exit the interactive chat session", "/exit")

        self.console.print(table)
        self.console.print("\n[dim]Tip: You can also ask plain English questions like 'What is this project about?' or 'Why did npm start fail?'[/dim]")

    def _cmd_check(self, target_file: Optional[str] = None) -> None:
        """Runs DocDrift verification pipeline."""
        from docdrift.cli import run_pipeline

        target = target_file if target_file else self.repo_path
        with self.console.status("[bold cyan]Running DocDrift ground-truth verification...[/bold cyan]", spinner="dots"):
            self.current_report = run_pipeline(
                target=target,
                model=self.model,
                offline=self.offline,
                probe_commands=self.probe_commands,
                prefer_ultsql=self.prefer_ultsql,
                cleanup_temp=False,
            )

        reporter = Reporter(console=self.console)
        reporter.print_terminal_report(self.current_report)

        if self.current_report.drifted_count > 0:
            self.console.print(
                f"\n[bold yellow]Found {self.current_report.drifted_count} drifted claim(s).[/bold yellow] "
                f"Type [bold green]/diff[/bold green] to review changes, [bold green]/fix[/bold green] to apply in-place, or [bold green]/pr[/bold green] for PR description."
            )

    def _cmd_test(self, arg: str = "") -> None:
        """Runs codebase health & execution verification."""
        from docdrift.verifiers.code_runner import CodeHealthVerifier
        from docdrift.cli import _display_health_report

        with self.console.status("[bold cyan]Verifying codebase syntax, entrypoints, dependencies & tests...[/bold cyan]", spinner="dots"):
            health_report = CodeHealthVerifier.check_codebase_health(self.repo_path, run_tests=True, test_timeout_sec=15.0)

        _display_health_report(health_report, self.console)

    def _cmd_drift(self) -> None:
        """Shows summary table of drifted claims."""
        if not self.current_report:
            self.console.print("[dim]No active scan report. Running /check first...[/dim]")
            self._cmd_check()
            return

        if self.current_report.drifted_count == 0:
            self.console.print("[bold green]✨ All claims verified! 0 drifted claims.[/bold green]")
            return

        table = Table(title="Drifted Claims & Required Fixes", border_style="red", expand=True)
        table.add_column("Location", style="cyan", width=22)
        table.add_column("Type", style="magenta", width=14)
        table.add_column("Outdated Line", style="red")
        table.add_column("Code Verification Proof", style="white")
        table.add_column("Proposed Fix", style="green")

        for claim in self.current_report.claims:
            if claim.verdict == Verdict.DRIFTED:
                loc = f"{claim.doc_path}:{claim.line_number}"
                fix = claim.fixed_line or claim.suggested_replacement or "⚠️ Needs human review"
                table.add_row(loc, claim.claim_type.value, claim.raw_line or claim.extracted_value, claim.evidence, fix)

        self.console.print(table)

    def _cmd_fix(self) -> None:
        """Applies code-constrained fixes in-place."""
        if not self.current_report:
            self.console.print("[dim]No active scan report. Running /check first...[/dim]")
            self._cmd_check()

        if not self.current_report or self.current_report.drifted_count == 0:
            self.console.print("[bold green]✨ No drift found. Nothing to fix![/bold green]")
            return

        reporter = Reporter(console=self.console)
        modified = reporter.apply_fixes(self.current_report)
        if modified:
            self.console.print(f"[bold green]✔ Successfully updated {len(modified)} file(s) in place:[/bold green]")
            for f in modified:
                rel = os.path.relpath(f, self.repo_path)
                self.console.print(f"  [green]•[/green] {rel}")
            # Re-run check to verify resolution
            self.console.print("[dim]Re-verifying repository...[/dim]")
            self._cmd_check()
        else:
            self.console.print("[yellow]No automated in-place fixes could be applied safely (flagged for maintainer review).[/yellow]")

    def _cmd_diff(self) -> None:
        """Displays unified diff of pending fixes."""
        if not self.current_report:
            self.console.print("[dim]No active scan report. Running /check first...[/dim]")
            self._cmd_check()

        if not self.current_report or self.current_report.drifted_count == 0:
            self.console.print("[bold green]✨ No drift found. No diff to display.[/bold green]")
            return

        self.console.print("[bold cyan]Git Unified Diff:[/bold cyan]")
        Reporter(console=self.console).print_diff(self.current_report)

    def _cmd_pr(self) -> None:
        """Displays GitHub PR description."""
        if not self.current_report:
            self.console.print("[dim]No active scan report. Running /check first...[/dim]")
            self._cmd_check()

        if not self.current_report or self.current_report.drifted_count == 0:
            self.console.print("[bold green]✨ No drift found. All documentation matches codebase perfectly.[/bold green]")
            return

        reporter = Reporter(console=self.console)
        desc = reporter.generate_pr_description(self.current_report, commit_sha=self.commit_sha or "HEAD")
        self.console.print(Panel(desc, title="[bold green]Ready-to-Merge GitHub PR Description[/bold green]", border_style="green"))

    def _cmd_verify(self, claim_text: str) -> None:
        """Verifies an ad-hoc claim or command on the fly."""
        if not claim_text:
            self.console.print("[yellow]Usage: /verify <command, file path, or env var>[/yellow]")
            return

        clean_text = claim_text.strip().strip("`")
        repo = Path(self.repo_path)

        # Detect claim type
        claim_type = ClaimType.COMMAND_SCRIPT
        if clean_text.isupper() and "_" in clean_text:
            claim_type = ClaimType.ENV_VAR
        elif clean_text.isdigit() and len(clean_text) in (4, 5):
            claim_type = ClaimType.DOCKER_PORT
        elif any(clean_text.endswith(ext) for ext in (".json", ".md", ".yml", ".yaml", ".py", ".ts", ".js", ".txt")) or "/" in clean_text or "\\" in clean_text:
            claim_type = ClaimType.FILE_PATH

        claim = Claim(
            doc_path="interactive_query",
            line_number=1,
            raw_line=clean_text,
            claim_type=claim_type,
            extracted_value=clean_text,
        )

        res: Optional[VerificationResult] = None
        if self.file_verifier.can_verify(claim):
            res = self.file_verifier.verify(claim, str(repo))
        elif self.cmd_verifier.can_verify(claim):
            res = self.cmd_verifier.verify(claim, str(repo))
            if res.verdict == Verdict.VERIFIED and self.sandbox_prober:
                probe_res = self.sandbox_prober.probe_command(claim.extracted_value, str(repo))
                if probe_res.verdict == Verdict.DRIFTED:
                    res.verdict = Verdict.DRIFTED
                    res.evidence = f"{res.evidence} | Probe: {probe_res.evidence}"
        elif self.dep_verifier.can_verify(claim):
            res = self.dep_verifier.verify(claim, str(repo))
        elif self.env_verifier.can_verify(claim):
            res = self.env_verifier.verify(claim, str(repo))

        if not res:
            res = VerificationResult(verdict=Verdict.UNKNOWN, evidence="Could not match claim with specialized verifier.")

        # Display result panel
        verdict_color = "green" if res.verdict == Verdict.VERIFIED else ("red" if res.verdict == Verdict.DRIFTED else "yellow")
        icon = "✔" if res.verdict == Verdict.VERIFIED else ("✗" if res.verdict == Verdict.DRIFTED else "?")

        table = Table.grid(expand=True)
        table.add_column(style="bold yellow", width=14)
        table.add_column()

        table.add_row("Claim:", f"[cyan]{clean_text}[/cyan] ({claim_type.value})")
        table.add_row("Verdict:", f"[{verdict_color}]{icon} {res.verdict.value.upper()}[/{verdict_color}]")
        table.add_row("Proof:", res.evidence)
        if res.suggested_value:
            table.add_row("Suggested Fix:", f"[bold green]{res.suggested_value}[/bold green]")

        self.console.print(Panel(table, title="[bold]Instant Codebase Verification[/bold]", border_style=verdict_color))

    def _cmd_files(self) -> None:
        """Lists discovered documentation files and manifests."""
        repo = Path(self.repo_path)
        table = Table(title="Repository Files & Manifests", border_style="cyan")
        table.add_column("Category", style="bold yellow")
        table.add_column("Path", style="cyan")

        for f in self.doc_files:
            rel = os.path.relpath(f, self.repo_path)
            table.add_row("Documentation", rel)

        manifest_names = ("package.json", "requirements.txt", "pyproject.toml", "docker-compose.yml", "Makefile", "Cargo.toml", "go.mod")
        for m in manifest_names:
            for p in repo.glob(f"**/{m}"):
                if not any(d in p.parts for d in (".git", "node_modules", ".venv")):
                    table.add_row("Manifest", p.relative_to(repo).as_posix())

        self.console.print(table)

    def _cmd_manifests(self) -> None:
        """Inspects declared scripts and targets."""
        repo = Path(self.repo_path)
        table = Table(title="Discovered Manifest Targets & Scripts", border_style="cyan")
        table.add_column("Ecosystem", style="bold yellow")
        table.add_column("Target / Script", style="green")
        table.add_column("Definition", style="dim")

        # Node
        node_scripts = self.cmd_verifier._get_node_scripts(repo)
        for k, v in node_scripts.items():
            table.add_row("Node / npm", k, v)

        # Python
        py_scripts = self.cmd_verifier._get_python_scripts(repo)
        for k, v in py_scripts.items():
            table.add_row("Python", k, v)

        # Make
        make_targets = self.cmd_verifier._get_make_targets(repo)
        for k in make_targets:
            table.add_row("Makefile", k, "")

        self.console.print(table)

    # --------------------------------------------------------------------------
    # Natural Language Conversational Q&A
    # --------------------------------------------------------------------------

    def _handle_natural_query(self, query: str) -> None:
        """Answers developer questions using Gemma 4 with codebase grounding."""
        # Check if user asks about running verification or fixing
        q_lower = query.lower()
        if any(w in q_lower for w in ("run check", "verify repo", "scan doc", "check repo", "check documentation")):
            self._cmd_check()
            return
        if any(w in q_lower for w in ("run test", "check health", "is code working", "test code", "code health")):
            self._cmd_test()
            return
        if any(w in q_lower for w in ("show diff", "what is the diff", "view diff", "git diff")):
            self._cmd_diff()
            return
        if any(w in q_lower for w in ("apply fix", "fix the drift", "fix doc", "auto fix", "fix it")):
            self._cmd_fix()
            return
        if any(w in q_lower for w in ("create pr", "generate pr", "pr description", "make pr")):
            self._cmd_pr()
            return

        with self.console.status("[bold cyan]DocDrift thinking with Gemma 4...[/bold cyan]", spinner="dots"):
            response_text = self._query_gemma(query)

        self.console.print()
        self.console.print(Markdown(response_text))

    def _query_gemma(self, query: str) -> str:
        """Queries Gemma 4 or heuristic fallback with codebase state."""
        # 1. Native GenAI multi-turn chat if online and active
        if self._genai_chat is not None:
            try:
                # Augment message with latest context if available
                context_parts = []
                if self.current_report and self.current_report.drifted_count > 0:
                    drift_summary = "; ".join(
                        f"{c.doc_path}:{c.line_number} '{c.extracted_value}' -> {c.evidence}"
                        for c in self.current_report.claims if c.verdict == Verdict.DRIFTED
                    )
                    context_parts.append(f"Current Drift Context: {drift_summary}")
                elif self.current_report and self.current_report.verified_count > 0:
                    context_parts.append(f"Current Scan Context: {self.current_report.verified_count} verified claims matching codebase.")

                # If asking what the project is about or general summary, attach primary README excerpt
                q_lower = query.lower()
                if any(w in q_lower for w in ("what is", "about", "overview", "purpose", "explain", "how does it work", "architecture")):
                    for doc in self.doc_files:
                        if os.path.basename(doc).lower() in ("readme.md", "readme"):
                            try:
                                with open(doc, "r", encoding="utf-8", errors="ignore") as f:
                                    snippet = f.read(3000).strip()
                                    if snippet:
                                        context_parts.append(f"Primary README excerpt:\n{snippet}")
                                        break
                            except Exception:
                                pass

                augmented_prompt = query
                if context_parts:
                    augmented_prompt = f"[{' | '.join(context_parts)}]\n\nUser Question: {query}"

                resp = self._genai_chat.send_message(augmented_prompt)
                if resp and resp.text:
                    return resp.text.strip()
            except Exception as e:
                pass

        # 2. Single-shot prompt via ai_client
        if not self.offline and not self.ai_client.is_offline and self.ai_client._client is not None:
            try:
                sys_inst = self._build_system_instruction()
                context_notes = []
                if self.current_report:
                    context_notes.append(
                        f"Current scan: {self.current_report.verified_count} verified, {self.current_report.drifted_count} drifted."
                    )
                    for c in self.current_report.claims:
                        if c.verdict == Verdict.DRIFTED:
                            context_notes.append(f"Drift at {c.doc_path}:{c.line_number}: {c.extracted_value} ({c.evidence})")

                # Attach README excerpt for project overview questions
                q_lower = query.lower()
                if any(w in q_lower for w in ("what is", "about", "overview", "purpose", "explain", "how does it work", "architecture")):
                    for doc in self.doc_files:
                        if os.path.basename(doc).lower() in ("readme.md", "readme"):
                            try:
                                with open(doc, "r", encoding="utf-8", errors="ignore") as f:
                                    snippet = f.read(3000).strip()
                                    if snippet:
                                        context_notes.append(f"Primary README excerpt:\n{snippet}")
                                        break
                            except Exception:
                                pass

                prompt = (
                    f"Repository Context: {self.repo_name}\n"
                    f"Status: {'; '.join(context_notes)}\n\n"
                    f"User Question: {query}\n\n"
                    f"Answer accurately based on the codebase facts and recommend /commands when helpful."
                )
                text = self.ai_client.generate_text(prompt=prompt, system_instruction=sys_inst)
                if text:
                    return text.strip()
            except Exception:
                pass

        # 3. Offline Heuristic Fallback
        return self._offline_answer(query)

    def _offline_answer(self, query: str) -> str:
        """Grounded offline response generation using repository AST and manifests."""
        q = query.lower()
        repo = Path(self.repo_path)

        # Questions about project purpose / what the project is about
        if any(w in q for w in ("what is", "about", "overview", "purpose", "describe", "explain")):
            for doc in self.doc_files:
                if os.path.basename(doc).lower() in ("readme.md", "readme"):
                    try:
                        with open(doc, "r", encoding="utf-8", errors="ignore") as f:
                            lines = []
                            for line in f:
                                s = line.strip()
                                if s and not s.startswith("#") and not s.startswith("[!") and not s.startswith("!"):
                                    lines.append(s)
                                    if len(lines) >= 6:
                                        break
                            if lines:
                                return f"**Project Overview ({self.repo_name}):**\n\n" + "\n\n".join(lines) + f"\n\nType `/check` to verify documentation claims or `/files` to view scanned docs."
                    except Exception:
                        pass

        # Questions about scripts or commands
        if any(w in q for w in ("script", "command", "npm", "poetry", "make", "start", "test", "dev")):
            node_scripts = self.cmd_verifier._get_node_scripts(repo)
            py_scripts = self.cmd_verifier._get_python_scripts(repo)
            make_targets = self.cmd_verifier._get_make_targets(repo)
            lines = ["**Discovered Repository Commands & Scripts:**\n"]
            if node_scripts:
                lines.append(f"- **Node / npm**: {', '.join(f'`npm run {k}`' for k in node_scripts.keys())}")
            if py_scripts:
                lines.append(f"- **Python**: {', '.join(f'`{k}`' for k in py_scripts.keys())}")
            if make_targets:
                lines.append(f"- **Makefile**: {', '.join(f'`make {k}`' for k in make_targets.keys())}")
            lines.append("\nYou can test any specific command on the fly with `/verify <command>`.")
            return "\n".join(lines)

        # Questions about drift status
        if any(w in q for w in ("drift", "outdated", "fail", "broken")):
            if self.current_report:
                if self.current_report.drifted_count == 0:
                    return "✨ **No documentation drift detected!** All documented claims match codebase manifests."
                items = [f"- **{c.doc_path}:{c.line_number}**: `{c.extracted_value}`\n  *Proof*: {c.evidence}" for c in self.current_report.claims if c.verdict == Verdict.DRIFTED]
                return f"**Detected Drift ({self.current_report.drifted_count} claims):**\n\n" + "\n".join(items) + "\n\nUse `/fix` to apply validated fixes or `/diff` to see the git diff."
            return "No verification scan has been run yet. Type `/check` to scan the repository documentation."

        # Questions about environment variables
        if any(w in q for w in ("env", "secret", "variable", "api_key", "token")):
            _, codebase_vars = self.env_verifier._scan_repository(repo, target_var="")
            lines = ["**Environment Variables Analysis:**\n"]
            if codebase_vars:
                lines.append(f"- Referenced in repository: {', '.join(f'`{v}`' for v in sorted(codebase_vars)[:15])}")
            lines.append("\nDocDrift confirms `.env.example` templates and ensures uncommitted `.env` files are never flagged as missing.")
            return "\n".join(lines)

        # Questions about documentation files
        if any(w in q for w in ("doc", "file", "readme", "markdown")):
            doc_rel = [os.path.relpath(f, self.repo_path) for f in self.doc_files]
            return (
                f"**Scanned Documentation Files ({len(doc_rel)}):**\n"
                + "\n".join(f"- `{f}`" for f in doc_rel)
                + "\n\nType `/check` to verify all claims in these files."
            )

        # General help
        return (
            f"I am DocDrift assistant grounded in **{os.path.basename(self.repo_path)}**.\n\n"
            f"- Run `/check` to scan documentation against codebase manifests.\n"
            f"- Run `/verify <command>` to test any command, file path, or script immediately.\n"
            f"- Run `/drift` to inspect current drifted claims and verification proofs.\n"
            f"- Run `/fix` to apply surgical code-constrained fixes in-place.\n"
            f"- Type `/help` for a complete list of commands."
        )

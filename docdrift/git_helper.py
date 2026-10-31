import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Optional, List, Tuple


class GitHelper:
    """Utilities for interacting with git repositories (local and remote)."""

    @staticmethod
    def is_git_url(target: str) -> bool:
        """Determines if the target string is a remote Git repository URL."""
        if not target:
            return False
        target = target.strip()
        if target.startswith(("https://github.com/", "http://github.com/", "git@github.com:")):
            return True
        if target.startswith(("https://", "http://", "git://", "ssh://")) and target.endswith(".git"):
            return True
        return False

    @staticmethod
    def clone_repo(git_url: str, depth: int = 1, max_retries: int = 2) -> Tuple[str, bool]:
        """Clones a remote repository into a temporary directory with retry resilience.

        Returns (path_to_cloned_repo, is_temporary).
        """
        temp_parent = tempfile.mkdtemp(prefix="docdrift_repo_")
        target_dir = os.path.join(temp_parent, "sandbox")
        cmd = ["git", "clone", "--depth", str(depth), git_url, target_dir]

        last_error = ""
        for attempt in range(1, max_retries + 1):
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if proc.returncode == 0 and os.path.isdir(target_dir):
                    return target_dir, True
                last_error = proc.stderr.strip() or proc.stdout.strip() or f"git exit code {proc.returncode}"
            except subprocess.TimeoutExpired:
                last_error = "git clone timed out after 60s"
            except Exception as e:
                last_error = str(e)

            if attempt < max_retries:
                shutil.rmtree(target_dir, ignore_errors=True)
                time.sleep(1.0)

        # Cleanup on total failure
        shutil.rmtree(temp_parent, ignore_errors=True)
        raise RuntimeError(f"Failed to clone repository '{git_url}': {last_error}")

    @staticmethod
    def cleanup_temp_repo(repo_path: str) -> None:
        """Safely removes a temporary cloned repository and its parent sandbox."""
        if not repo_path:
            return
        parent = os.path.dirname(repo_path)
        if "docdrift_repo_" in os.path.basename(parent):
            shutil.rmtree(parent, ignore_errors=True)
        elif "docdrift_repo_" in os.path.basename(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)

    @staticmethod
    def get_commit_hash(repo_path: str) -> str:
        """Returns the current short commit hash or 'unknown'."""
        try:
            cmd = ["git", "-C", repo_path, "rev-parse", "--short", "HEAD"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except Exception:
            return "local-dev"

    @staticmethod
    def pull_updates(repo_path: str) -> Tuple[bool, str]:
        """Pulls remote updates (git pull --ff-only).

        Returns (has_new_updates, message).
        """
        try:
            cmd = ["git", "-C", repo_path, "pull", "--ff-only"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            output = result.stdout.strip()
            if "Already up to date." in output:
                return False, output
            return True, output
        except subprocess.TimeoutExpired:
            return False, "Git pull timed out."
        except Exception as e:
            return False, f"Git pull failed: {e}"

    @staticmethod
    def create_branch_and_commit(
        repo_path: str,
        branch_name: str,
        commit_message: str,
        modified_files: List[str]
    ) -> bool:
        """Creates a new branch and commits modified files."""
        try:
            # Create and switch to new branch
            subprocess.run(
                ["git", "-C", repo_path, "checkout", "-b", branch_name],
                capture_output=True, text=True, check=True
            )
            # Add modified files
            for f in modified_files:
                subprocess.run(
                    ["git", "-C", repo_path, "add", f],
                    capture_output=True, text=True, check=True
                )
            # Commit
            subprocess.run(
                ["git", "-C", repo_path, "commit", "-m", commit_message],
                capture_output=True, text=True, check=True
            )
            return True
        except Exception:
            return False

    @staticmethod
    def find_repo_root(start_path: Optional[str] = None) -> str:
        """Auto-detects the repository or project root by traversing upwards."""
        from pathlib import Path
        curr = Path(start_path or os.getcwd()).resolve()
        for parent in [curr] + list(curr.parents):
            if (parent / ".git").is_dir() or (parent / ".git").is_file():
                return str(parent)
            if any((parent / f).exists() for f in (
                "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile", "docker-compose.yml"
            )):
                return str(parent)
            if any((parent / f).exists() for f in ("README.md", "README", "readme.md")):
                return str(parent)
        return str(curr)

    @staticmethod
    def detect_ecosystem(repo_path: str) -> Tuple[List[str], bool]:
        """Detects languages, frameworks, and whether the repository is a monorepo."""
        from pathlib import Path
        repo = Path(repo_path)
        ecosystems = []
        is_monorepo = False

        if any(repo.glob("*.py")) or any(repo.glob("*/*.py")) or (repo / "requirements.txt").exists() or (repo / "pyproject.toml").exists():
            ecosystems.append("Python")
        if (repo / "package.json").exists() or any(repo.glob("*/package.json")):
            ecosystems.append("Node.js/TypeScript")
        if (repo / "Cargo.toml").exists():
            ecosystems.append("Rust")
        if (repo / "go.mod").exists():
            ecosystems.append("Go")
        if (repo / "Dockerfile").exists() or (repo / "docker-compose.yml").exists():
            ecosystems.append("Docker")

        if (repo / "backend").is_dir() and (repo / "frontend").is_dir():
            is_monorepo = True
        elif any((repo / d).is_dir() for d in ("packages", "apps")):
            is_monorepo = True

        return ecosystems or ["General"], is_monorepo

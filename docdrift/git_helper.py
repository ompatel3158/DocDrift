import os
import re
import shutil
import subprocess
import tempfile
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
    def clone_repo(git_url: str, depth: int = 1) -> Tuple[str, bool]:
        """Clones a remote repository into a temporary directory.

        Returns (path_to_cloned_repo, is_temporary).
        """
        temp_dir = tempfile.mkdtemp(prefix="docdrift_repo_")
        try:
            cmd = ["git", "clone", "--depth", str(depth), git_url, temp_dir]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return temp_dir, True
        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to clone repository '{git_url}': {e}")

    @staticmethod
    def cleanup_temp_repo(repo_path: str) -> None:
        """Safely removes a temporary cloned repository."""
        if repo_path and os.path.exists(repo_path) and "docdrift_repo_" in repo_path:
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

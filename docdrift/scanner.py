import hashlib
import os
import re
from pathlib import Path
from typing import List, Set
from docdrift.models import DocSection


# Directories to always ignore during document scanning
DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
    "coverage",
    ".pytest_cache",
    "__pycache__",
    ".next",
    ".nuxt",
    ".docdrift",
}


class DocScanner:
    """Discovers documentation files (README, docs/*.md) and splits them into sections."""

    def __init__(self, repo_path: str, custom_globs: List[str] = None):
        self.repo_path = os.path.abspath(repo_path)
        self.custom_globs = custom_globs or []

    def find_doc_files(self) -> List[str]:
        """Finds all relevant markdown documents in the repository."""
        doc_files: List[str] = []

        if not os.path.exists(self.repo_path):
            return doc_files

        # If repo_path is itself a file
        if os.path.isfile(self.repo_path):
            if self.repo_path.lower().endswith((".md", ".markdown", ".rst")):
                return [self.repo_path]
            return []

        # Find README file variations
        readme_candidates = [
            "README.md", "readme.md", "Readme.md",
            "README.markdown", "README.rst"
        ]
        for name in readme_candidates:
            candidate = os.path.join(self.repo_path, name)
            if os.path.isfile(candidate) and candidate not in doc_files:
                doc_files.append(candidate)

        # Walk through the repo finding markdown files while respecting ignores
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".")]

            for file in files:
                if file.lower().endswith((".md", ".markdown")):
                    full_path = os.path.join(root, file)
                    if full_path not in doc_files:
                        doc_files.append(full_path)

        # Sort for deterministic processing (README first, then docs)
        def sort_key(path: str):
            rel = os.path.relpath(path, self.repo_path)
            if "readme" in rel.lower():
                return (0, rel)
            if rel.startswith("docs"):
                return (1, rel)
            return (2, rel)

        doc_files.sort(key=sort_key)
        return doc_files

    def parse_sections(self, file_path: str) -> List[DocSection]:
        """Splits a markdown document into logical sections based on headers (#, ##, ###).

        Preserves 1-based start line numbers and computes content SHA-256 hashes.
        """
        if not os.path.isfile(file_path):
            return []

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return []

        lines = content.splitlines()
        if not lines:
            return []

        sections: List[DocSection] = []
        current_title = "Overview"
        current_start_line = 1
        current_lines: List[str] = []

        header_regex = re.compile(r"^(#{1,6})\s+(.+)$")

        for idx, line in enumerate(lines, start=1):
            match = header_regex.match(line.strip())
            if match and current_lines:
                # Flush the preceding section
                section_text = "\n".join(current_lines)
                sections.append(
                    DocSection(
                        doc_path=file_path,
                        section_title=current_title,
                        start_line=current_start_line,
                        lines=list(current_lines),
                        content_hash=hashlib.sha256(section_text.encode("utf-8")).hexdigest(),
                    )
                )
                current_title = match.group(2).strip()
                current_start_line = idx
                current_lines = [line]
            else:
                if match and not current_lines:
                    current_title = match.group(2).strip()
                    current_start_line = idx
                current_lines.append(line)

        # Flush the final section
        if current_lines:
            section_text = "\n".join(current_lines)
            sections.append(
                DocSection(
                    doc_path=file_path,
                    section_title=current_title,
                    start_line=current_start_line,
                    lines=current_lines,
                    content_hash=hashlib.sha256(section_text.encode("utf-8")).hexdigest(),
                )
            )

        return sections

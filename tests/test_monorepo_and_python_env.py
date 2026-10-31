"""Tests for monorepo subprojects, Python modules (pip, venv), ASGI runners, and .env handling."""

import os
from pathlib import Path
import pytest

from docdrift.models import Claim, ClaimType, Verdict
from docdrift.verifiers.command_verifier import MultiManifestVerifier
from docdrift.verifiers.file_verifier import FileVerifier


@pytest.fixture
def monorepo(tmp_path: Path) -> Path:
    """Creates a sample monorepo with backend (FastAPI/python) and frontend (React/Vite)."""
    repo = tmp_path / "sample_monorepo"
    repo.mkdir()

    # Root README and gitignore
    (repo / ".gitignore").write_text(".env\n.venv/\nnode_modules/\n", encoding="utf-8")
    (repo / "README.md").write_text("# Monorepo\n\nSee `backend/README.md`\n", encoding="utf-8")

    # Backend
    backend = repo / "backend"
    backend.mkdir()
    (backend / ".env.example").write_text("API_KEY=test\nPORT=8000\n", encoding="utf-8")
    (backend / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
    (backend / "README.md").write_text("cd backend\n", encoding="utf-8")

    app_dir = backend / "app"
    app_dir.mkdir()
    (app_dir / "__init__.py").write_text("", encoding="utf-8")
    (app_dir / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")

    # Frontend
    frontend = repo / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text(
        '{\n  "name": "frontend",\n  "scripts": {\n    "dev": "vite",\n    "build": "vite build"\n  }\n}',
        encoding="utf-8"
    )

    return repo


def test_env_file_verified_when_template_exists(monorepo: Path):
    """FileVerifier marks .env verified if .env.example exists and doc instructs setup/security."""
    verifier = FileVerifier()
    claim = Claim(
        doc_path=str(monorepo / "backend" / "README.md"),
        line_number=5,
        raw_line="Never commit `.env`.",
        claim_type=ClaimType.FILE_PATH,
        extracted_value=".env",
    )
    res = verifier.verify(claim, str(monorepo))
    assert res.verdict == Verdict.VERIFIED
    assert "omitted from git repository" in res.evidence


def test_python_pip_requirements_verified(monorepo: Path):
    """MultiManifestVerifier verifies python -m pip with existing requirements.txt."""
    verifier = MultiManifestVerifier()
    claim = Claim(
        doc_path=str(monorepo / "backend" / "README.md"),
        line_number=4,
        raw_line="python -m pip install -r requirements.txt",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="python -m pip install -r requirements.txt",
    )
    res = verifier.verify(claim, str(monorepo))
    assert res.verdict == Verdict.VERIFIED
    assert "requirements.txt" in res.evidence


def test_python_venv_command_verified(monorepo: Path):
    """py -m venv or python -m venv is recognized as standard python module."""
    verifier = MultiManifestVerifier()
    claim = Claim(
        doc_path=str(monorepo / "backend" / "README.md"),
        line_number=2,
        raw_line="py -m venv .venv",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="py -m venv .venv",
    )
    res = verifier.verify(claim, str(monorepo))
    assert res.verdict == Verdict.VERIFIED
    assert "venv" in res.evidence


def test_uvicorn_asgi_entrypoint_verified(monorepo: Path):
    """uvicorn app.main:app is verified against backend/app/main.py."""
    verifier = MultiManifestVerifier()
    claim = Claim(
        doc_path=str(monorepo / "backend" / "README.md"),
        line_number=7,
        raw_line="uvicorn app.main:app --reload --port 8000",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="uvicorn app.main:app --reload --port 8000",
    )
    res = verifier.verify(claim, str(monorepo))
    assert res.verdict == Verdict.VERIFIED
    assert "app.main:app" in res.evidence


def test_cd_directory_verified(monorepo: Path):
    """cd backend and cd frontend verify against existing directories."""
    verifier = MultiManifestVerifier()
    claim = Claim(
        doc_path=str(monorepo / "README.md"),
        line_number=1,
        raw_line="cd backend",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="cd backend",
    )
    res = verifier.verify(claim, str(monorepo))
    assert res.verdict == Verdict.VERIFIED
    assert "backend" in res.evidence


def test_copy_item_template_verified(monorepo: Path):
    """Copy-Item .env.example .env verifies that .env.example exists."""
    verifier = MultiManifestVerifier()
    claim = Claim(
        doc_path=str(monorepo / "backend" / "README.md"),
        line_number=5,
        raw_line="Copy-Item .env.example .env",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="Copy-Item .env.example .env",
    )
    res = verifier.verify(claim, str(monorepo))
    assert res.verdict == Verdict.VERIFIED
    assert ".env.example" in res.evidence


def test_monorepo_frontend_npm_scripts_verified(monorepo: Path):
    """npm run dev verifies against frontend/package.json in monorepo."""
    verifier = MultiManifestVerifier()
    claim = Claim(
        doc_path=str(monorepo / "README.md"),
        line_number=10,
        raw_line="npm run dev",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run dev",
    )
    res = verifier.verify(claim, str(monorepo))
    assert res.verdict == Verdict.VERIFIED
    assert "dev" in res.evidence

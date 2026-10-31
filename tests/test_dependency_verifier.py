"""Unit tests for DependencyVerifier."""

from pathlib import Path
import pytest

from docdrift.models import Claim, ClaimType, Verdict
from docdrift.verifiers.dependency_verifier import DependencyVerifier


class TestDependencyVerifier:
    def setup_method(self):
        self.verifier = DependencyVerifier()

    def test_get_declared_python_dependencies(self, tmp_path: Path):
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            "fastapi>=0.68.0\n"
            "uvicorn[standard]==0.15.0\n"
            "pyyaml\n"
            "# comment\n"
            "requests\n",
            encoding="utf-8",
        )
        deps = self.verifier.get_declared_python_dependencies(tmp_path)
        assert "fastapi" in deps
        assert "uvicorn" in deps
        assert "pyyaml" in deps
        assert "yaml" in deps  # mapped import
        assert "requests" in deps

    def test_verify_python_requirements_all_satisfied(self, tmp_path: Path):
        # Create requirements.txt
        (tmp_path / "requirements.txt").write_text("fastapi\nuvicorn\n", encoding="utf-8")
        # Create source code using fastapi
        src_dir = tmp_path / "app"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("import fastapi\napp = fastapi.FastAPI()\n", encoding="utf-8")

        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="pip install -r requirements.txt",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="pip install -r requirements.txt",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED
        assert "All source code imports are declared" in res.evidence

    def test_verify_python_requirements_missing_import_detected_as_drift(self, tmp_path: Path):
        # Create requirements.txt missing redis
        (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
        # Source code imports redis
        (tmp_path / "worker.py").write_text("import fastapi\nimport redis\n", encoding="utf-8")

        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="pip install -r requirements.txt",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="pip install -r requirements.txt",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert "redis" in res.evidence
        assert "missing dependencies" in res.evidence.lower()

    def test_verify_pip_package_install_verified(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="pip install fastapi",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="pip install fastapi",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED

    def test_verify_pip_package_install_drifted_when_bogus(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="pip install bogus-nonexistent-library",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="pip install bogus-nonexistent-library",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert "neither declared" in res.evidence

    def test_verify_node_dependencies_all_satisfied(self, tmp_path: Path):
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text('{\n  "dependencies": {\n    "express": "^4.18.0"\n  }\n}', encoding="utf-8")
        (tmp_path / "server.js").write_text("const express = require('express');\n", encoding="utf-8")

        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="npm install",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="npm install",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED
        assert "All source code imports are declared" in res.evidence

    def test_verify_node_dependencies_missing_detected_as_drift(self, tmp_path: Path):
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text('{\n  "dependencies": {\n    "express": "^4.18.0"\n  }\n}', encoding="utf-8")
        (tmp_path / "server.js").write_text("const express = require('express');\nconst lodash = require('lodash');\n", encoding="utf-8")

        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="npm install",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="npm install",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert "lodash" in res.evidence

    def test_transitive_framework_dependencies_satisfied(self, tmp_path: Path):
        # fastapi transitively provides pydantic and starlette
        (tmp_path / "requirements.txt").write_text("fastapi==0.115.12\n", encoding="utf-8")
        src_dir = tmp_path / "app"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("from pydantic import BaseModel\nimport starlette\n", encoding="utf-8")

        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="pip install -r requirements.txt",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="pip install -r requirements.txt",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED
        assert "All source code imports are declared" in res.evidence

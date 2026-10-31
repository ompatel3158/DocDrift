"""Tests for DocDrift deterministic verifiers."""

import os
from pathlib import Path
import pytest

from docdrift.models import Claim, ClaimType, Verdict
from docdrift.verifiers.file_verifier import FileVerifier
from docdrift.verifiers.command_verifier import MultiManifestVerifier
from docdrift.verifiers.env_verifier import EnvVerifier


SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_drifted_repo"


# ==============================================================================
# FileVerifier Tests
# ==============================================================================

class TestFileVerifier:
    def setup_method(self):
        self.verifier = FileVerifier()

    def test_can_verify(self):
        claim_file = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Check `src/index.ts`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="src/index.ts",
        )
        claim_cmd = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Run `npm test`",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="npm test",
        )
        assert self.verifier.can_verify(claim_file) is True
        assert self.verifier.can_verify(claim_cmd) is False

    def test_verify_existing_file_relative_to_root(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Source: `src/index.ts`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="src/index.ts",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED
        assert "exists in repository" in res.evidence

    def test_verify_existing_directory(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Directory: `config`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="config",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED
        assert res.metadata.get("is_dir") is True

    def test_verify_existing_file_in_subdirectory(self):
        claim = Claim(
            doc_path="README.md",
            line_number=20,
            raw_line="Settings in `config/settings.json`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="config/settings.json",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED

    def test_verify_missing_file_with_close_match(self):
        # Claim claims config/database.json, repo has config/settings.json
        claim = Claim(
            doc_path="README.md",
            line_number=20,
            raw_line="Settings in `config/database.json`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="config/database.json",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.DRIFTED
        assert res.suggested_value is not None
        assert "settings.json" in res.suggested_value

    def test_verify_relative_to_doc_directory(self, tmp_path):
        # Create repo with docs/guide.md pointing to ../src/app.py
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "app.py").write_text("print('hello')")
        (docs_dir / "guide.md").write_text("[App](../src/app.py)")

        claim = Claim(
            doc_path="docs/guide.md",
            line_number=1,
            raw_line="See [App](../src/app.py)",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="../src/app.py",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED

    def test_verify_path_with_markdown_anchor(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="See `src/index.ts#L10`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="src/index.ts#L10",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED

    def test_verify_completely_nonexistent_path(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Check `nonexistent/deep/missing.txt`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="nonexistent/deep/missing.txt",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.DRIFTED


# ==============================================================================
# MultiManifestVerifier Tests
# ==============================================================================

class TestMultiManifestVerifier:
    def setup_method(self):
        self.verifier = MultiManifestVerifier()

    def test_can_verify(self):
        c1 = Claim(doc_path="README.md", line_number=1, raw_line="run", claim_type=ClaimType.COMMAND_SCRIPT, extracted_value="npm test")
        c2 = Claim(doc_path="README.md", line_number=1, raw_line="port", claim_type=ClaimType.DOCKER_PORT, extracted_value="8080")
        c3 = Claim(doc_path="README.md", line_number=1, raw_line="env", claim_type=ClaimType.ENV_VAR, extracted_value="PORT")
        assert self.verifier.can_verify(c1) is True
        assert self.verifier.can_verify(c2) is True
        assert self.verifier.can_verify(c3) is False

    # --- Node / JS Ecosystem ---
    def test_node_npm_test_verified(self):
        claim = Claim(
            doc_path="README.md",
            line_number=15,
            raw_line="npm test",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="npm test",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED
        assert "jest" in res.evidence

    def test_node_npm_run_build_verified(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="npm run build",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="npm run build",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED

    def test_node_drifted_script_suggests_match(self):
        # sample_drifted_repo has 'test', 'build', 'start'. README says 'npm run dev'
        claim = Claim(
            doc_path="README.md",
            line_number=10,
            raw_line="npm run dev",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="npm run dev",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.DRIFTED
        assert "package.json" in res.evidence
        assert res.suggested_value is not None

    def test_node_missing_package_json(self, tmp_path):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="npm run build",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="npm run build",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert "requires 'package.json'" in res.evidence

    # --- Python Ecosystem ---
    def test_python_poetry_run_verified(self, tmp_path):
        pyproj = tmp_path / "pyproject.toml"
        pyproj.write_text("""
[tool.poetry.scripts]
serve = "my_app.server:main"
worker = "my_app.worker:run"
""")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="poetry run serve",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="poetry run serve",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED
        assert "pyproject.toml" in res.evidence

    def test_python_poetry_run_drifted(self, tmp_path):
        pyproj = tmp_path / "pyproject.toml"
        pyproj.write_text("""
[tool.poetry.scripts]
serve = "my_app.server:main"
""")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="poetry run server",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="poetry run server",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert res.suggested_value == "poetry run serve"

    def test_python_uv_run_verified(self, tmp_path):
        pyproj = tmp_path / "pyproject.toml"
        pyproj.write_text("""
[project.scripts]
docdrift = "docdrift.cli:main"
""")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="uv run docdrift",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="uv run docdrift",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED

    def test_python_module_run_verified(self, tmp_path):
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "cli.py").write_text("print('cli')")

        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="python -m mypkg.cli",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="python -m mypkg.cli",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED

    # --- Docker Ecosystem ---
    def test_docker_compose_service_verified(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="docker compose up api",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="docker compose up api",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED
        assert "api" in res.evidence

    def test_docker_compose_service_drifted(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="docker compose up api_server",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="docker compose up api_server",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.DRIFTED
        assert res.suggested_value == "docker compose up api"

    def test_docker_port_verified(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Exposed on port 8080",
            claim_type=ClaimType.DOCKER_PORT,
            extracted_value="8080",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED
        assert "8080" in res.evidence

    def test_docker_port_drifted(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Exposed on port 9000",
            claim_type=ClaimType.DOCKER_PORT,
            extracted_value="9000",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.DRIFTED
        assert "Declared ports: 8080:8080" in res.evidence

    def test_dockerfile_expose_verified(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM alpine\nEXPOSE 3000\n")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Exposed port 3000",
            claim_type=ClaimType.DOCKER_PORT,
            extracted_value="3000",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED

    # --- Rust Ecosystem ---
    def test_rust_cargo_run_verified(self, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("""
[package]
name = "my_crate"
version = "0.1.0"

[[bin]]
name = "server"
path = "src/server.rs"
""")
        claim1 = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="cargo run",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="cargo run",
        )
        claim2 = Claim(
            doc_path="README.md",
            line_number=2,
            raw_line="cargo run --bin server",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="cargo run --bin server",
        )
        assert self.verifier.verify(claim1, str(tmp_path)).verdict == Verdict.VERIFIED
        assert self.verifier.verify(claim2, str(tmp_path)).verdict == Verdict.VERIFIED

    def test_rust_cargo_drifted_bin(self, tmp_path):
        cargo = tmp_path / "Cargo.toml"
        cargo.write_text("""
[package]
name = "my_crate"
version = "0.1.0"

[[bin]]
name = "server"
path = "src/server.rs"
""")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="cargo run --bin servr",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="cargo run --bin servr",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert res.suggested_value == "cargo run --bin server"

    # --- Go Ecosystem ---
    def test_go_run_verified(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/app\n\ngo 1.21\n")
        (tmp_path / "main.go").write_text("package main\nfunc main() {}\n")

        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="go run main.go",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="go run main.go",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED

    def test_go_missing_source_file(self, tmp_path):
        (tmp_path / "go.mod").write_text("module example.com/app\n")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="go run missing.go",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="go run missing.go",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED

    # --- Make Ecosystem ---
    def test_make_target_verified(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="make build",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="make build",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED
        assert "Makefile" in res.evidence

    def test_make_target_drifted(self):
        # sample_drifted_repo has build, clean, test, deploy. README claims 'make lint'
        claim = Claim(
            doc_path="README.md",
            line_number=30,
            raw_line="make lint",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="make lint",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.DRIFTED
        assert "Available targets" in res.evidence
        assert res.suggested_value is not None


# ==============================================================================
# EnvVerifier Tests
# ==============================================================================

class TestEnvVerifier:
    def setup_method(self):
        self.verifier = EnvVerifier()

    def test_can_verify(self):
        claim_env = Claim(doc_path="README.md", line_number=1, raw_line="PORT", claim_type=ClaimType.ENV_VAR, extracted_value="PORT")
        claim_file = Claim(doc_path="README.md", line_number=1, raw_line="file", claim_type=ClaimType.FILE_PATH, extracted_value="README.md")
        assert self.verifier.can_verify(claim_env) is True
        assert self.verifier.can_verify(claim_file) is False

    def test_env_var_verified_in_sample_repo(self):
        # PORT exists in .env.example, docker-compose.yml, and src/index.ts
        claim = Claim(
            doc_path="README.md",
            line_number=24,
            raw_line="Server port: `PORT`",
            claim_type=ClaimType.ENV_VAR,
            extracted_value="PORT",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED
        assert "PORT" in res.evidence
        assert len(res.metadata["files"]) >= 2

    def test_env_var_database_url_verified(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="DATABASE_URL",
            claim_type=ClaimType.ENV_VAR,
            extracted_value="DATABASE_URL",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.VERIFIED

    def test_env_var_drifted_suggests_similar_name(self):
        # README claims MONGO_URL, but repo uses DATABASE_URL
        claim = Claim(
            doc_path="README.md",
            line_number=23,
            raw_line="Mongo URL: `MONGO_URL`",
            claim_type=ClaimType.ENV_VAR,
            extracted_value="MONGO_URL",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.DRIFTED
        assert res.suggested_value is not None
        assert "DATABASE_URL" in res.suggested_value

    def test_env_var_port_num_drifted_suggests_port(self):
        # README claims PORT_NUM, but repo uses PORT
        claim = Claim(
            doc_path="README.md",
            line_number=24,
            raw_line="`PORT_NUM`",
            claim_type=ClaimType.ENV_VAR,
            extracted_value="PORT_NUM",
        )
        res = self.verifier.verify(claim, str(SAMPLE_REPO))
        assert res.verdict == Verdict.DRIFTED
        assert res.suggested_value == "PORT"

    def test_env_var_go_and_rust_patterns(self, tmp_path):
        go_file = tmp_path / "app.go"
        go_file.write_text('os.Getenv("SECRET_JWT_KEY")')
        rs_file = tmp_path / "main.rs"
        rs_file.write_text('env::var("APP_DATABASE_URL")')

        c1 = Claim(doc_path="README.md", line_number=1, raw_line="jwt", claim_type=ClaimType.ENV_VAR, extracted_value="SECRET_JWT_KEY")
        c2 = Claim(doc_path="README.md", line_number=1, raw_line="db", claim_type=ClaimType.ENV_VAR, extracted_value="APP_DATABASE_URL")

        assert self.verifier.verify(c1, str(tmp_path)).verdict == Verdict.VERIFIED
        assert self.verifier.verify(c2, str(tmp_path)).verdict == Verdict.VERIFIED

    def test_env_var_completely_absent(self, tmp_path):
        (tmp_path / ".env.example").write_text("FOO=bar\n")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="UNKNOWN_VAR_XYZ",
            claim_type=ClaimType.ENV_VAR,
            extracted_value="UNKNOWN_VAR_XYZ",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert res.suggested_value is None

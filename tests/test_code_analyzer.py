"""Unit tests for CodeAnalyzer AST parsing, syntax validation, and symbol extraction."""

from pathlib import Path
import pytest

from docdrift.verifiers.code_analyzer import CodeAnalyzer


class TestCodeAnalyzerSyntax:
    def test_python_valid_syntax(self, tmp_path: Path):
        code_file = tmp_path / "valid.py"
        code_file.write_text("def hello(name: str) -> str:\n    return f'Hello {name}'\n", encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(code_file)
        assert is_valid is True
        assert err is None

    def test_python_invalid_syntax(self, tmp_path: Path):
        code_file = tmp_path / "syntax_err.py"
        code_file.write_text("def broken_func(\n    return 42\n", encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(code_file)
        assert is_valid is False
        assert err is not None
        assert "SyntaxError" in err
        assert "line" in err

    def test_json_valid_syntax(self, tmp_path: Path):
        json_file = tmp_path / "valid.json"
        json_file.write_text('{"name": "test", "active": true, "ports": [80, 443]}', encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(json_file)
        assert is_valid is True
        assert err is None

    def test_json_invalid_syntax(self, tmp_path: Path):
        json_file = tmp_path / "broken.json"
        json_file.write_text('{"name": "test", unquoted_key: 123,}', encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(json_file)
        assert is_valid is False
        assert err is not None
        assert "JSON" in err

    def test_yaml_valid_syntax(self, tmp_path: Path):
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text("service:\n  name: web\n  ports:\n    - 8080:8080\n", encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(yaml_file)
        assert is_valid is True
        assert err is None

    def test_yaml_invalid_syntax(self, tmp_path: Path):
        yaml_file = tmp_path / "broken.yaml"
        yaml_file.write_text("service:\n  name: [unclosed list\n  ports: : bad\n", encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(yaml_file)
        assert is_valid is False
        assert err is not None
        assert "YAML" in err

    def test_javascript_bracket_balance(self, tmp_path: Path):
        js_file = tmp_path / "broken.js"
        js_file.write_text("function test() {\n    if (true) {\n        console.log('unclosed');\n}\n", encoding="utf-8")
        is_valid, err = CodeAnalyzer.validate_file_syntax(js_file)
        assert is_valid is False
        assert err is not None
        assert "Unclosed bracket" in err or "bracket" in err.lower()


class TestCodeAnalyzerSymbols:
    def test_extract_python_symbols(self, tmp_path: Path):
        py_file = tmp_path / "main.py"
        py_file.write_text(
            "from fastapi import FastAPI\n"
            "import os\n"
            "app = FastAPI()\n"
            "server: int = 100\n"
            "def handler(req):\n"
            "    pass\n"
            "async def fetch_data():\n"
            "    pass\n"
            "class Worker:\n"
            "    pass\n"
            "__all__ = ['exported_func']\n",
            encoding="utf-8",
        )
        symbols = CodeAnalyzer.extract_declared_symbols(py_file)
        assert "app" in symbols
        assert "server" in symbols
        assert "handler" in symbols
        assert "fetch_data" in symbols
        assert "Worker" in symbols
        assert "FastAPI" in symbols
        assert "exported_func" in symbols

    def test_verify_symbol_in_file_found(self, tmp_path: Path):
        py_file = tmp_path / "app.py"
        py_file.write_text("app = 'fastapi_instance'\n", encoding="utf-8")
        is_decl, all_syms, suggestion = CodeAnalyzer.verify_symbol_in_file(py_file, "app")
        assert is_decl is True
        assert suggestion is None

    def test_verify_symbol_in_file_missing_with_close_match(self, tmp_path: Path):
        py_file = tmp_path / "app.py"
        py_file.write_text("application = 'fastapi_instance'\n", encoding="utf-8")
        is_decl, all_syms, suggestion = CodeAnalyzer.verify_symbol_in_file(py_file, "app")
        assert is_decl is False
        assert suggestion == "application"


class TestCodeAnalyzerImports:
    def test_extract_python_third_party_imports(self, tmp_path: Path):
        py_file = tmp_path / "service.py"
        py_file.write_text(
            "import os\n"
            "import sys\n"
            "import json\n"
            "from pathlib import Path\n"
            "import fastapi\n"
            "from pydantic import BaseModel\n"
            "import requests\n",
            encoding="utf-8",
        )
        imports = CodeAnalyzer.extract_imports_from_file(py_file)
        # Standard library modules (os, sys, json, pathlib) should be excluded
        assert "os" not in imports
        assert "sys" not in imports
        assert "json" not in imports
        assert "pathlib" not in imports
        # Third-party packages should be included
        assert "fastapi" in imports
        assert "pydantic" in imports
        assert "requests" in imports

    def test_extract_node_imports(self, tmp_path: Path):
        js_file = tmp_path / "index.js"
        js_file.write_text(
            "const fs = require('fs');\n"
            "const path = require('node:path');\n"
            "const express = require('express');\n"
            "import lodash from 'lodash';\n"
            "import { get } from '@types/node';\n"
            "const local = require('./local_module');\n",
            encoding="utf-8",
        )
        imports = CodeAnalyzer.extract_imports_from_file(js_file)
        # Node standard library and relative paths should be excluded
        assert "fs" not in imports
        assert "path" not in imports
        assert "./local_module" not in imports
        # Third-party packages should be included
        assert "express" in imports
        assert "lodash" in imports
        assert "@types/node" in imports

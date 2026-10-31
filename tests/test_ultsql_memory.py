"""Comprehensive unit and integration tests for DocDrift's UltSQL & Local Memory Backends."""

import os
import shutil
import tempfile
from typing import Generator
import pytest

from docdrift.memory import (
    BaseMemoryBackend,
    LocalMemoryBackend,
    UltSQLMCPClient,
    UltSQLMemoryBackend,
    compute_symbol_embedding,
    cosine_similarity,
    get_memory_backend,
)
from docdrift.models import Claim, ClaimType, Verdict
from docdrift.verifiers.env_verifier import EnvVerifier


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    path = tempfile.mkdtemp(prefix="docdrift_test_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# =============================================================================
# 1. Embedding & Tokenization Utility Tests
# =============================================================================

def test_symbol_embedding_determinism():
    vec1 = compute_symbol_embedding("MONGO_URI")
    vec2 = compute_symbol_embedding("MONGO_URI")
    assert len(vec1) == 64
    assert vec1 == vec2
    assert all(isinstance(x, float) for x in vec1)


def test_symbol_embedding_semantic_clustering():
    vec_mongo = compute_symbol_embedding("MONGO_URI")
    vec_db = compute_symbol_embedding("DATABASE_URL")
    vec_port = compute_symbol_embedding("PORT")
    vec_secret = compute_symbol_embedding("API_SECRET_KEY")

    sim_db = cosine_similarity(vec_mongo, vec_db)
    sim_port = cosine_similarity(vec_mongo, vec_port)
    sim_secret = cosine_similarity(vec_mongo, vec_secret)

    # MONGO_URI and DATABASE_URL share strong semantic synonym cluster
    assert sim_db > 0.70
    assert sim_db > sim_port
    assert sim_db > sim_secret


def test_symbol_embedding_port_tokens():
    vec_port_num = compute_symbol_embedding("PORT_NUM")
    vec_port = compute_symbol_embedding("PORT")
    vec_db = compute_symbol_embedding("DATABASE_URL")

    sim_port = cosine_similarity(vec_port_num, vec_port)
    sim_db = cosine_similarity(vec_port_num, vec_db)

    assert sim_port > 0.70
    assert sim_port > sim_db


# =============================================================================
# 2. LocalMemoryBackend Tests (Zero-Dependency SQLite & Vector Fallback)
# =============================================================================

def test_local_backend_kv_file_hash(temp_dir: str):
    backend = LocalMemoryBackend(repo_path=temp_dir)
    assert backend.get_file_hash("README.md") is None

    backend.set_file_hash("README.md", "hash_abc_123")
    assert backend.get_file_hash("README.md") == "hash_abc_123"

    # Overwrite
    backend.set_file_hash("README.md", "hash_def_456")
    assert backend.get_file_hash("README.md") == "hash_def_456"

    # Normalized path check
    assert backend.get_file_hash("./README.md") == "hash_def_456"
    backend.close()


def test_local_backend_claims_history(temp_dir: str):
    backend = LocalMemoryBackend(repo_path=temp_dir)

    c1 = Claim(
        doc_path="README.md",
        line_number=10,
        raw_line="npm run dev",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run dev",
        verdict=Verdict.DRIFTED,
        evidence="Script dev not found in package.json",
        suggested_replacement="npm start",
    )
    c2 = Claim(
        doc_path="README.md",
        line_number=25,
        raw_line="MONGO_URL=mongodb://localhost",
        claim_type=ClaimType.ENV_VAR,
        extracted_value="MONGO_URL",
        verdict=Verdict.DRIFTED,
        evidence="DATABASE_URL exists in .env.example",
        suggested_replacement="DATABASE_URL",
    )
    c3 = Claim(
        doc_path="docs/setup.md",
        line_number=5,
        raw_line="PORT=3000",
        claim_type=ClaimType.ENV_VAR,
        extracted_value="PORT",
        verdict=Verdict.VERIFIED,
        evidence="Found in .env.example",
    )

    backend.store_claim(c1)
    backend.store_claim(c2)
    backend.store_claim(c3)

    # Fetch all
    all_claims = backend.get_claims()
    assert len(all_claims) == 3

    # Filter by verdict
    drifted_claims = backend.get_claims({"verdict": Verdict.DRIFTED})
    assert len(drifted_claims) == 2
    assert {c.id for c in drifted_claims} == {c1.id, c2.id}

    # Filter by claim_type
    env_claims = backend.get_claims({"claim_type": ClaimType.ENV_VAR})
    assert len(env_claims) == 2
    assert {c.id for c in env_claims} == {c2.id, c3.id}

    # Filter by doc_path
    setup_claims = backend.get_claims({"doc_path": "docs/setup.md"})
    assert len(setup_claims) == 1
    assert setup_claims[0].id == c3.id

    backend.close()


def test_local_backend_manifest_scripts(temp_dir: str):
    backend = LocalMemoryBackend(repo_path=temp_dir)
    scripts = [
        {"ecosystem": "npm", "manifest_path": "package.json", "script_name": "build", "command_str": "tsc"},
        {"ecosystem": "npm", "manifest_path": "package.json", "script_name": "test", "command_str": "jest"},
        {"ecosystem": "make", "manifest_path": "Makefile", "script_name": "lint", "command_str": "flake8"},
    ]
    backend.index_manifest_scripts(scripts)

    stats = backend.get_stats()
    assert stats["manifest_scripts_count"] == 3
    backend.close()


def test_local_backend_vector_similarity(temp_dir: str):
    backend = LocalMemoryBackend(repo_path=temp_dir)
    symbols = [
        {"file_path": ".env.example", "symbol_name": "DATABASE_URL", "symbol_type": "env_var"},
        {"file_path": ".env.example", "symbol_name": "PORT", "symbol_type": "env_var"},
        {"file_path": ".env.example", "symbol_name": "REDIS_HOST", "symbol_type": "env_var"},
    ]
    backend.index_symbols(symbols)

    # Query for MONGO_URI
    matches = backend.find_similar_symbol("MONGO_URI", symbol_type="env_var", limit=2)
    assert len(matches) == 2
    assert matches[0]["symbol_name"] == "DATABASE_URL"
    assert matches[0]["similarity"] > 0.70

    # Query for PORT_NUM
    port_matches = backend.find_similar_symbol("PORT_NUM", symbol_type="env_var", limit=2)
    assert len(port_matches) == 2
    assert port_matches[0]["symbol_name"] == "PORT"
    assert port_matches[0]["similarity"] > 0.70

    backend.close()


def test_local_backend_clear_and_stats(temp_dir: str):
    backend = LocalMemoryBackend(repo_path=temp_dir)
    backend.set_file_hash("test.txt", "123")
    backend.store_claim(Claim(
        doc_path="doc.md",
        line_number=1,
        raw_line="val",
        claim_type=ClaimType.CLI_FLAG,
        extracted_value="val"
    ))

    stats_before = backend.get_stats()
    assert stats_before["cached_file_hashes"] == 1
    assert stats_before["claims_history_count"] == 1

    backend.clear()
    stats_after = backend.get_stats()
    assert stats_after["cached_file_hashes"] == 0
    assert stats_after["claims_history_count"] == 0
    backend.close()


# =============================================================================
# 3. UltSQL Converged Memory Backend Tests (KV, NoSQL, Relational, HNSW)
# =============================================================================

@pytest.mark.skipif(not UltSQLMemoryBackend.is_available(), reason="UltSQL is not available on this machine")
def test_ultsql_backend_kv_file_hash(temp_dir: str):
    backend = UltSQLMemoryBackend(repo_path=temp_dir)
    assert backend.get_file_hash("README.md") is None

    backend.set_file_hash("README.md", "hash_ult_123")
    assert backend.get_file_hash("README.md") == "hash_ult_123"

    backend.set_file_hash("README.md", "hash_ult_456")
    assert backend.get_file_hash("README.md") == "hash_ult_456"
    backend.close()


@pytest.mark.skipif(not UltSQLMemoryBackend.is_available(), reason="UltSQL is not available on this machine")
def test_ultsql_backend_claims_history(temp_dir: str):
    backend = UltSQLMemoryBackend(repo_path=temp_dir)

    c1 = Claim(
        doc_path="README.md",
        line_number=10,
        raw_line="npm run dev",
        claim_type=ClaimType.COMMAND_SCRIPT,
        extracted_value="npm run dev",
        verdict=Verdict.DRIFTED,
        evidence="Missing dev in package.json",
        suggested_replacement="npm start",
    )
    c2 = Claim(
        doc_path="README.md",
        line_number=23,
        raw_line="MONGO_URL=...",
        claim_type=ClaimType.ENV_VAR,
        extracted_value="MONGO_URL",
        verdict=Verdict.DRIFTED,
        evidence="DATABASE_URL found in .env.example",
        suggested_replacement="DATABASE_URL",
    )
    c3 = Claim(
        doc_path="docs/guide.md",
        line_number=5,
        raw_line="PORT=3000",
        claim_type=ClaimType.ENV_VAR,
        extracted_value="PORT",
        verdict=Verdict.VERIFIED,
        evidence="Matches .env.example",
    )

    backend.store_claim(c1)
    backend.store_claim(c2)
    backend.store_claim(c3)

    # Get all claims
    all_claims = backend.get_claims()
    assert len(all_claims) == 3

    # Filter by verdict
    drifted_claims = backend.get_claims({"verdict": Verdict.DRIFTED})
    assert len(drifted_claims) == 2
    assert {c.id for c in drifted_claims} == {c1.id, c2.id}

    # Filter by claim_type
    env_claims = backend.get_claims({"claim_type": ClaimType.ENV_VAR})
    assert len(env_claims) == 2
    assert {c.id for c in env_claims} == {c2.id, c3.id}

    backend.close()


@pytest.mark.skipif(not UltSQLMemoryBackend.is_available(), reason="UltSQL is not available on this machine")
def test_ultsql_backend_manifest_scripts(temp_dir: str):
    backend = UltSQLMemoryBackend(repo_path=temp_dir)
    scripts = [
        {"id": "sc1", "ecosystem": "npm", "manifest_path": "package.json", "script_name": "build", "command_str": "tsc"},
        {"id": "sc2", "ecosystem": "npm", "manifest_path": "package.json", "script_name": "test", "command_str": "jest"},
    ]
    backend.index_manifest_scripts(scripts)

    stats = backend.get_stats()
    assert stats["manifest_scripts_count"] == 2
    backend.close()


@pytest.mark.skipif(not UltSQLMemoryBackend.is_available(), reason="UltSQL is not available on this machine")
def test_ultsql_backend_hnsw_vector_rag(temp_dir: str):
    backend = UltSQLMemoryBackend(repo_path=temp_dir)
    symbols = [
        {"id": "s1", "file_path": ".env.example", "symbol_name": "DATABASE_URL", "symbol_type": "env_var"},
        {"id": "s2", "file_path": ".env.example", "symbol_name": "PORT", "symbol_type": "env_var"},
        {"id": "s3", "file_path": ".env.example", "symbol_name": "REDIS_HOST", "symbol_type": "env_var"},
    ]
    backend.index_symbols(symbols)

    # 1. Query for MONGO_URI
    matches = backend.find_similar_symbol("MONGO_URI", symbol_type="env_var", limit=2)
    assert len(matches) >= 1
    assert matches[0]["symbol_name"] == "DATABASE_URL"
    assert matches[0]["similarity"] > 0.70

    # 2. Query for PORT_NUM
    port_matches = backend.find_similar_symbol("PORT_NUM", symbol_type="env_var", limit=2)
    assert len(port_matches) >= 1
    assert port_matches[0]["symbol_name"] == "PORT"
    assert port_matches[0]["similarity"] > 0.70

    backend.close()


@pytest.mark.skipif(not UltSQLMemoryBackend.is_available(), reason="UltSQL is not available on this machine")
def test_ultsql_backend_clear_and_stats(temp_dir: str):
    backend = UltSQLMemoryBackend(repo_path=temp_dir)
    backend.set_file_hash("doc.md", "hash123")
    backend.index_manifest_scripts([
        {"id": "s1", "ecosystem": "npm", "manifest_path": "package.json", "script_name": "dev", "command_str": "vite"}
    ])

    stats = backend.get_stats()
    assert stats["cached_file_hashes"] == 1
    assert stats["manifest_scripts_count"] == 1

    backend.clear()
    stats_cleared = backend.get_stats()
    assert stats_cleared["cached_file_hashes"] == 0
    assert stats_cleared["manifest_scripts_count"] == 0
    backend.close()


# =============================================================================
# 4. Factory & Diagnostics Tests
# =============================================================================

def test_factory_prefer_ultsql(temp_dir: str):
    backend = get_memory_backend(repo_path=temp_dir, prefer_ultsql=True)
    if UltSQLMemoryBackend.is_available():
        assert isinstance(backend, UltSQLMemoryBackend)
        diag = backend.get_diagnostics()
        assert diag["engine"] == "UltSQL Converged Multimodal Database"
    else:
        assert isinstance(backend, LocalMemoryBackend)
    backend.close()


def test_factory_fallback_to_local(temp_dir: str):
    backend = get_memory_backend(repo_path=temp_dir, prefer_ultsql=False)
    assert isinstance(backend, LocalMemoryBackend)
    stats = backend.get_stats()
    assert stats["engine"] == "local_sqlite"
    backend.close()


def test_ultsql_diagnostics():
    client = UltSQLMCPClient()
    diag = client.get_diagnostics()
    assert "status" in diag
    assert "running" in diag
    client.close()


# =============================================================================
# 5. Integration with Sample Drifted Repository & EnvVerifier
# =============================================================================

def test_integration_with_env_verifier_and_memory(temp_dir: str):
    sample_repo = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixtures",
        "sample_drifted_repo",
    )
    if not os.path.exists(sample_repo):
        pytest.skip("sample_drifted_repo fixture not found")

    # Use memory backend initialized with repository
    backend = get_memory_backend(repo_path=sample_repo, prefer_ultsql=True)

    verifier = EnvVerifier(memory=backend)

    # Claim in README: MONGO_URL (drifted, .env.example has DATABASE_URL)
    claim_mongo = Claim(
        doc_path="README.md",
        line_number=23,
        raw_line="- `MONGO_URL`: Connection string for MongoDB",
        claim_type=ClaimType.ENV_VAR,
        extracted_value="MONGO_URL",
    )
    res_mongo = verifier.verify(claim_mongo, repo_path=sample_repo)
    assert res_mongo.verdict == Verdict.DRIFTED
    assert res_mongo.suggested_value == "DATABASE_URL"

    # Claim in README: PORT_NUM (drifted, .env.example has PORT)
    claim_port = Claim(
        doc_path="README.md",
        line_number=24,
        raw_line="- `PORT_NUM`: Server port number",
        claim_type=ClaimType.ENV_VAR,
        extracted_value="PORT_NUM",
    )
    res_port = verifier.verify(claim_port, repo_path=sample_repo)
    assert res_port.verdict == Verdict.DRIFTED
    assert res_port.suggested_value == "PORT"

    backend.close()

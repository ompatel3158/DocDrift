"""Base memory backend interface and embedding utilities for DocDrift."""

from abc import ABC, abstractmethod
import math
import re
from typing import Any, Dict, List, Optional
import zlib

from docdrift.models import Claim


# Semantic synonym clusters for codebase symbols and environment variables
SYNONYM_GROUPS = [
    {
        "mongo",
        "mongodb",
        "database",
        "db",
        "sql",
        "postgres",
        "postgresql",
        "mysql",
        "sqlite",
        "nosql",
        "storage",
        "store",
        "datasource",
        "dsn",
        "connection",
    },
    {
        "uri",
        "url",
        "endpoint",
        "address",
        "addr",
        "host",
        "hostname",
        "link",
    },
    {
        "port",
        "port_num",
        "listen",
        "listen_port",
        "socket",
        "bind",
    },
    {
        "password",
        "pass",
        "pwd",
        "secret",
        "token",
        "key",
        "auth",
        "jwt",
        "credential",
        "access_token",
        "api_key",
    },
    {
        "env",
        "environment",
        "config",
        "settings",
        "options",
        "conf",
        "cfg",
    },
    {
        "user",
        "username",
        "usr",
        "login",
        "account",
    },
    {
        "path",
        "dir",
        "directory",
        "folder",
        "root",
        "file",
        "filepath",
    },
    {
        "server",
        "service",
        "app",
        "application",
        "backend",
        "api",
        "daemon",
    },
    {
        "test",
        "spec",
        "check",
        "build",
        "compile",
        "bundle",
        "start",
        "run",
        "serve",
        "dev",
        "develop",
    },
]


def tokenize_symbol(s: str) -> List[str]:
    """Tokenize camelCase, snake_case, kebab-case, or dot-separated symbol."""
    words = re.findall(r"[A-Za-z0-9]+", s)
    tokens: List[str] = []
    for w in words:
        split_camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", w).split()
        for x in split_camel:
            tokens.append(x.lower())
    return tokens


def compute_symbol_embedding(s: str, dim: int = 64) -> List[float]:
    """Deterministic, zero-dependency symbol embedding using tokenization,

    synonym expansion, character trigram feature hashing, and L2 normalization.
    """
    vec = [0.0] * dim
    tokens = tokenize_symbol(s)
    if not tokens:
        return vec

    # 1. Direct tokens (weight: 3.0)
    for t in tokens:
        idx = zlib.crc32(f"token:{t}".encode("utf-8")) % dim
        vec[idx] += 3.0

    # 2. Semantic synonym cluster expansion (weight: 1.5)
    for t in tokens:
        for group in SYNONYM_GROUPS:
            if t in group:
                for syn in group:
                    if syn != t:
                        idx = zlib.crc32(f"token:{syn}".encode("utf-8")) % dim
                        vec[idx] += 1.5

    # 3. Character trigrams for typo/suffix/prefix resilience (weight: 0.8)
    s_clean = re.sub(r"[^a-zA-Z0-9]", "", s).lower()
    for i in range(len(s_clean) - 2):
        ng = s_clean[i : i + 3]
        idx = zlib.crc32(f"ng:{ng}".encode("utf-8")) % dim
        vec[idx] += 0.8

    # 4. L2 normalization
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [round(x / norm, 6) for x in vec]
    return vec


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two unit vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


class BaseMemoryBackend(ABC):
    """Abstract base class for DocDrift persistent memory engines."""

    @abstractmethod
    def get_file_hash(self, path: str) -> Optional[str]:
        """Retrieve the cached SHA-256/content hash for a given file path."""
        pass

    @abstractmethod
    def set_file_hash(self, path: str, hash_val: str) -> None:
        """Store the SHA-256/content hash for a given file path."""
        pass

    @abstractmethod
    def store_claim(self, claim: Claim) -> None:
        """Persist a verified or drifted claim into memory history."""
        pass

    @abstractmethod
    def get_claims(self, filters: Optional[Dict[str, Any]] = None) -> List[Claim]:
        """Query claims from history with optional key/value filters."""
        pass

    @abstractmethod
    def index_manifest_scripts(self, scripts: List[Dict[str, Any]]) -> None:
        """Index package/manifest build, test, and run scripts into relational storage.

        Each script dict should contain: id, ecosystem, manifest_path,
        script_name, command_str.
        """
        pass

    @abstractmethod
    def index_symbols(self, symbols: List[Dict[str, Any]]) -> None:
        """Index code symbols and environment variables with vector embeddings.

        Each symbol dict should contain: id, file_path, symbol_name, symbol_type.
        """
        pass

    @abstractmethod
    def find_similar_symbol(
        self, query: str, symbol_type: str = "env_var", limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Find semantically similar symbols using vector RAG nearest neighbor search."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Persist any uncommitted state to disk."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Purge all stored hashes, claims, scripts, and symbols."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics and item counts from memory engine."""
        pass

    def close(self) -> None:
        """Close backend connections and release resources."""
        self.save()

    def __enter__(self) -> "BaseMemoryBackend":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

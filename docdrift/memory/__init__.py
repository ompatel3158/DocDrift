"""DocDrift Converged Memory & Vector RAG Subsystem."""

import logging
from typing import Optional

from docdrift.memory.base import (
    BaseMemoryBackend,
    compute_symbol_embedding,
    cosine_similarity,
)
from docdrift.memory.local_backend import LocalMemoryBackend
from docdrift.memory.ultsql_backend import UltSQLMCPClient, UltSQLMemoryBackend

logger = logging.getLogger("docdrift.memory")


def get_memory_backend(
    repo_path: str = ".",
    prefer_ultsql: bool = True,
    db_path: Optional[str] = None,
) -> BaseMemoryBackend:
    """Factory function to instantiate a DocDrift memory backend.

    Parameters:
        repo_path: Path to the target local repository.
        prefer_ultsql: If True, attempts to connect to UltSQL converged engine
                       (Relational + NoSQL + KV + HNSW Vector RAG).
        db_path: Optional explicit database path/directory or ':memory:'.

    Returns:
        An initialized BaseMemoryBackend (either UltSQLMemoryBackend or LocalMemoryBackend fallback).
    """
    if prefer_ultsql:
        try:
            if UltSQLMemoryBackend.is_available():
                logger.info("Connecting to UltSQL Converged Memory Backend...")
                return UltSQLMemoryBackend(repo_path=repo_path, db_path=db_path)
            else:
                logger.warning(
                    "UltSQL is requested but not available. Falling back to LocalMemoryBackend."
                )
        except Exception as e:
            logger.warning(
                f"Failed to initialize UltSQL backend ({e}). Falling back to LocalMemoryBackend."
            )

    return LocalMemoryBackend(repo_path=repo_path, db_path=db_path)


__all__ = [
    "BaseMemoryBackend",
    "LocalMemoryBackend",
    "UltSQLMemoryBackend",
    "UltSQLMCPClient",
    "get_memory_backend",
    "compute_symbol_embedding",
    "cosine_similarity",
]

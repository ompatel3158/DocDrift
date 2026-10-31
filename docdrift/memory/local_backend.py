"""Zero-dependency SQLite & local vector memory backend for DocDrift."""

import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional
import uuid

from docdrift.memory.base import (
    BaseMemoryBackend,
    compute_symbol_embedding,
    cosine_similarity,
)
from docdrift.models import Claim, ClaimType, Verdict


class LocalMemoryBackend(BaseMemoryBackend):
    """Zero-dependency local memory engine using SQLite and in-process vector math.

    Stores file hashes, claims history, manifest scripts, and symbol embeddings.
    Provides offline semantic fallback when UltSQL is unavailable.
    """

    def __init__(self, repo_path: str = ".", db_path: Optional[str] = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        if db_path:
            self.db_path = db_path
        elif repo_path == ":memory:":
            self.db_path = ":memory:"
        else:
            docdrift_dir = os.path.join(self.repo_path, ".docdrift")
            os.makedirs(docdrift_dir, exist_ok=True)
            self.db_path = os.path.join(docdrift_dir, "memory.db")

        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,  # autocommit mode
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS file_hashes (
                        path TEXT PRIMARY KEY,
                        hash_val TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS claims_history (
                        id TEXT PRIMARY KEY,
                        doc_path TEXT NOT NULL,
                        line_number INTEGER NOT NULL,
                        raw_line TEXT NOT NULL,
                        section_title TEXT,
                        claim_type TEXT NOT NULL,
                        extracted_value TEXT NOT NULL,
                        verdict TEXT NOT NULL,
                        evidence TEXT DEFAULT '',
                        suggested_replacement TEXT,
                        fixed_line TEXT,
                        metadata_json TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS manifest_scripts (
                        id TEXT PRIMARY KEY,
                        ecosystem TEXT NOT NULL,
                        manifest_path TEXT NOT NULL,
                        script_name TEXT NOT NULL,
                        command_str TEXT NOT NULL
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS repo_symbols (
                        id TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        symbol_name TEXT NOT NULL,
                        symbol_type TEXT NOT NULL,
                        embedding_json TEXT NOT NULL
                    );
                """)

    def _norm_path(self, path: str) -> str:
        return os.path.normpath(path).replace("\\", "/")

    def get_file_hash(self, path: str) -> Optional[str]:
        norm = self._norm_path(path)
        with self._lock:
            conn = self._get_connection()
            cur = conn.execute("SELECT hash_val FROM file_hashes WHERE path = ?", (norm,))
            row = cur.fetchone()
            return row["hash_val"] if row else None

    def set_file_hash(self, path: str, hash_val: str) -> None:
        norm = self._norm_path(path)
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "INSERT OR REPLACE INTO file_hashes (path, hash_val) VALUES (?, ?)",
                (norm, hash_val),
            )

    def store_claim(self, claim: Claim) -> None:
        data = claim.model_dump(mode="json")
        meta_str = json.dumps(data.get("metadata", {}))
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT OR REPLACE INTO claims_history (
                    id, doc_path, line_number, raw_line, section_title,
                    claim_type, extracted_value, verdict, evidence,
                    suggested_replacement, fixed_line, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.id,
                    self._norm_path(claim.doc_path),
                    claim.line_number,
                    claim.raw_line,
                    claim.section_title,
                    claim.claim_type.value if hasattr(claim.claim_type, "value") else str(claim.claim_type),
                    claim.extracted_value,
                    claim.verdict.value if hasattr(claim.verdict, "value") else str(claim.verdict),
                    claim.evidence,
                    claim.suggested_replacement,
                    claim.fixed_line,
                    meta_str,
                ),
            )

    def get_claims(self, filters: Optional[Dict[str, Any]] = None) -> List[Claim]:
        query = "SELECT * FROM claims_history"
        params: List[Any] = []
        if filters:
            clauses: List[str] = []
            for k, v in filters.items():
                val = v.value if hasattr(v, "value") else v
                if k == "doc_path" and isinstance(val, str):
                    val = self._norm_path(val)
                clauses.append(f"{k} = ?")
                params.append(val)
            if clauses:
                query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY rowid ASC"
        with self._lock:
            conn = self._get_connection()
            cur = conn.execute(query, params)
            rows = cur.fetchall()

        claims: List[Claim] = []
        for r in rows:
            meta = {}
            if r["metadata_json"]:
                try:
                    meta = json.loads(r["metadata_json"])
                except Exception:
                    meta = {}
            claim_dict = {
                "id": r["id"],
                "doc_path": r["doc_path"],
                "line_number": r["line_number"],
                "raw_line": r["raw_line"],
                "section_title": r["section_title"],
                "claim_type": ClaimType(r["claim_type"]),
                "extracted_value": r["extracted_value"],
                "verdict": Verdict(r["verdict"]),
                "evidence": r["evidence"] or "",
                "suggested_replacement": r["suggested_replacement"],
                "fixed_line": r["fixed_line"],
                "metadata": meta,
            }
            claims.append(Claim.model_validate(claim_dict))
        return claims

    def index_manifest_scripts(self, scripts: List[Dict[str, Any]]) -> None:
        if not scripts:
            return
        with self._lock:
            conn = self._get_connection()
            for s in scripts:
                s_id = str(s.get("id") or uuid.uuid4().hex[:12])
                ecosystem = str(s.get("ecosystem", "generic"))
                manifest_path = self._norm_path(str(s.get("manifest_path", "")))
                script_name = str(s.get("script_name", ""))
                command_str = str(s.get("command_str", ""))
                conn.execute(
                    """
                    INSERT OR REPLACE INTO manifest_scripts (
                        id, ecosystem, manifest_path, script_name, command_str
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (s_id, ecosystem, manifest_path, script_name, command_str),
                )

    def index_symbols(self, symbols: List[Dict[str, Any]]) -> None:
        if not symbols:
            return
        with self._lock:
            conn = self._get_connection()
            for sym in symbols:
                sym_id = str(sym.get("id") or uuid.uuid4().hex[:12])
                file_path = self._norm_path(str(sym.get("file_path", "")))
                symbol_name = str(sym.get("symbol_name", ""))
                symbol_type = str(sym.get("symbol_type", "env_var"))

                vec = sym.get("embedding")
                if not vec or not isinstance(vec, list):
                    vec = compute_symbol_embedding(symbol_name)

                embedding_json = json.dumps(vec)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO repo_symbols (
                        id, file_path, symbol_name, symbol_type, embedding_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (sym_id, file_path, symbol_name, symbol_type, embedding_json),
                )

    def find_similar_symbol(
        self, query: str, symbol_type: str = "env_var", limit: int = 3
    ) -> List[Dict[str, Any]]:
        query_vec = compute_symbol_embedding(query)
        sql = "SELECT id, file_path, symbol_name, symbol_type, embedding_json FROM repo_symbols"
        params: List[Any] = []
        if symbol_type:
            sql += " WHERE symbol_type = ?"
            params.append(symbol_type)

        with self._lock:
            conn = self._get_connection()
            cur = conn.execute(sql, params)
            rows = cur.fetchall()

        candidates: List[Dict[str, Any]] = []
        for r in rows:
            try:
                emb = json.loads(r["embedding_json"])
                sim = cosine_similarity(query_vec, emb)
                dist = max(0.0, 1.0 - sim)
                candidates.append({
                    "id": r["id"],
                    "file_path": r["file_path"],
                    "symbol_name": r["symbol_name"],
                    "symbol_type": r["symbol_type"],
                    "similarity": round(sim, 4),
                    "distance": round(dist, 4),
                })
            except Exception:
                continue

        # Sort descending by similarity (ascending by distance)
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:limit]

    def save(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute("DELETE FROM file_hashes;")
            conn.execute("DELETE FROM claims_history;")
            conn.execute("DELETE FROM manifest_scripts;")
            conn.execute("DELETE FROM repo_symbols;")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._get_connection()
            hashes = conn.execute("SELECT COUNT(*) FROM file_hashes").fetchone()[0]
            claims = conn.execute("SELECT COUNT(*) FROM claims_history").fetchone()[0]
            scripts = conn.execute("SELECT COUNT(*) FROM manifest_scripts").fetchone()[0]
            symbols = conn.execute("SELECT COUNT(*) FROM repo_symbols").fetchone()[0]

        return {
            "engine": "local_sqlite",
            "db_path": self.db_path,
            "cached_file_hashes": hashes,
            "claims_history_count": claims,
            "manifest_scripts_count": scripts,
            "indexed_symbols_count": symbols,
        }

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

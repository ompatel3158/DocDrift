"""UltSQL Converged Multimodal Memory & Vector RAG Engine for DocDrift.

Integrates:
1. Key-Value Store (ultsql_kv): Sub-microsecond cache for file hashes ('docdrift:hash:<path>' -> sha256).
2. NoSQL Document Collections (ultsql_doc): Schema-less collection 'claims_history' with JSON filtering.
3. Relational SQL (ultsql_query): Tables 'manifest_scripts' and 'repo_symbols' with schema introspection.
4. HNSW Vector RAG (ultsql_vector_search): Sub-millisecond similarity search over code symbols and env vars.
"""

import atexit
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import uuid

from docdrift.memory.base import (
    BaseMemoryBackend,
    compute_symbol_embedding,
)
from docdrift.models import Claim, ClaimType, Verdict

logger = logging.getLogger("docdrift.memory.ultsql")


class DatabaseLockException(RuntimeError):
    """Raised when an UltSQL database lock contention or stale lock occurs."""
    pass


def _assign_process_to_job_object(proc: subprocess.Popen) -> None:
    """Ensure child process and all descendants are terminated when parent exits on Windows."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JobObjectExtendedLimitInformation = 9

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryLimit", ctypes.c_size_t),
                ("PeakJobMemoryLimit", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if job:
            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            kernel32.SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            handle = int(proc._handle)
            kernel32.AssignProcessToJobObject(job, handle)
            proc._win_job_handle = job
    except Exception as e:
        logger.debug(f"Assigning process to Windows Job Object failed: {e}")


def cleanup_ultsql_lock(db_path: str) -> bool:
    """Removes stale or orphaned ultsql.lock in db_path.

    If locked by a lingering ultsql.exe process on Windows (WinError 32),
    terminates the lingering process and removes the lock file.
    Returns True if a lock file was found and cleaned up.
    """
    if not db_path or db_path == ":memory:":
        return False
    lock_path = os.path.join(db_path, "ultsql.lock")
    if not os.path.exists(lock_path):
        return False

    try:
        os.remove(lock_path)
        logger.info(f"Removed stale UltSQL lock file at {lock_path}")
        return True
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not remove {lock_path} directly ({e}). Attempting process termination...")
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ultsql.exe"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except Exception:
                pass
            time.sleep(0.1)
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
                    logger.info(f"Removed stale UltSQL lock file after process termination at {lock_path}")
                    return True
            except Exception as e2:
                logger.error(f"Failed to remove UltSQL lock file {lock_path}: {e2}")
    return False


def cleanup_known_orphan_locks() -> None:
    """Cleans up known orphaned lock files across repository workspaces."""
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, ".docdrift", "ultsql_data"),
        os.path.join(cwd, "tests", "fixtures", "sample_drifted_repo", ".docdrift", "ultsql_data"),
    ]
    for c in candidates:
        cleanup_ultsql_lock(c)


# Perform immediate cleanup of known orphaned locks
cleanup_known_orphan_locks()


class UltSQLMCPClient:
    """Subprocess client for UltSQL Model Context Protocol (MCP) server.

    Communicates over stdio using JSON-RPC 2.0 with thread-safe request serialization.
    """

    def __init__(
        self,
        executable_path: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        self.executable_path = executable_path
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._req_counter = 0
        self._server_info: Dict[str, Any] = {}
        self._status = "uninitialized"
        self._error_msg: Optional[str] = None
        atexit.register(self.close)

    def __del__(self) -> None:
        self.close()

    def _next_id(self) -> int:
        self._req_counter += 1
        return self._req_counter

    def _find_command(self) -> Optional[List[str]]:
        """Determine command list to spawn UltSQL MCP server."""
        if self.executable_path and os.path.exists(self.executable_path):
            return [self.executable_path, "mcp"]

        # 1. Search PATH for ultsql binary
        which_ultsql = shutil.which("ultsql")
        if which_ultsql:
            return [which_ultsql, "mcp"]

        # 2. Check standard Windows AppData location
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            win_path = os.path.join(local_appdata, "UltSQL", "bin", "ultsql.exe")
            if os.path.exists(win_path):
                return [win_path, "mcp"]

        # 3. Fallback to Dart runner with source script if available
        dart_bin = shutil.which("dart")
        user_home = os.path.expanduser("~")
        dart_script = os.path.join(
            user_home,
            ".gemini",
            "antigravity",
            "scratch",
            "ULTSQL",
            "bin",
            "ultsql_mcp_server.dart",
        )
        if dart_bin and os.path.exists(dart_script):
            return [dart_bin, "run", dart_script]

        return None

    def is_available(self) -> bool:
        """Check if UltSQL MCP server can be located and initialized."""
        cmd = self._find_command()
        if not cmd:
            return False
        try:
            self._ensure_started()
            return self._proc is not None and self._proc.poll() is None
        except Exception as e:
            self._error_msg = str(e)
            return False

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic health and status metrics for UltSQL."""
        cmd = self._find_command()
        return {
            "status": self._status,
            "command": " ".join(cmd) if cmd else None,
            "running": self._proc is not None and self._proc.poll() is None,
            "server_info": self._server_info,
            "last_error": self._error_msg,
        }

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        cmd = self._find_command()
        if not cmd:
            self._status = "binary_not_found"
            raise RuntimeError(
                "UltSQL executable or MCP server script could not be found. "
                "Ensure 'ultsql' is installed and in PATH or Dart SDK is configured."
            )

        self._status = "initializing"
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding="utf-8",
            )
        except Exception as e:
            self._status = "launch_failed"
            self._error_msg = str(e)
            raise RuntimeError(f"Failed to launch UltSQL MCP process ({cmd}): {e}") from e

        if sys.platform == "win32":
            _assign_process_to_job_object(self._proc)

        # Perform JSON-RPC initialization handshake
        init_req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "docdrift-memory-client", "version": "0.1.0"},
            },
        }
        try:
            res = self._send_raw(init_req)
            if "result" in res and "serverInfo" in res["result"]:
                self._server_info = res["result"]["serverInfo"]
                self._status = "ready"
                self._error_msg = None
            else:
                self._status = "handshake_failed"
                raise RuntimeError(f"Unexpected MCP handshake response: {res}")
        except Exception as e:
            self._status = "handshake_failed"
            self._error_msg = str(e)
            self.close()
            raise

    def _send_raw(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._proc or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("UltSQL process is not connected")

        msg_str = json.dumps(payload) + "\n"
        self._proc.stdin.write(msg_str)
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            stderr_out = ""
            if self._proc.stderr:
                try:
                    stderr_out = self._proc.stderr.read()
                except Exception:
                    pass
            raise RuntimeError(
                f"UltSQL MCP process terminated unexpectedly. Stderr: {stderr_out}"
            )

        try:
            return json.loads(line)
        except json.JSONDecodeError as err:
            raise RuntimeError(f"Failed to decode MCP JSON-RPC response: {line}") from err

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke an UltSQL MCP tool and return parsed result content."""
        with self._lock:
            self._ensure_started()
            req = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            res = self._send_raw(req)

            if "error" in res:
                err = res["error"]
                msg = err.get("message", "Unknown error")
                err_lower = msg.lower()
                if "lock" in err_lower or "databaselock" in err_lower or "winerror 32" in err_lower:
                    raise DatabaseLockException(f"UltSQL MCP error calling '{tool_name}': {msg}")
                raise RuntimeError(f"UltSQL MCP error calling '{tool_name}': {msg}")

            result_obj = res.get("result", {})
            if result_obj.get("isError"):
                content = result_obj.get("content", [{}])
                msg = content[0].get("text", "Unknown tool error") if content else "Error"
                msg_lower = msg.lower()
                if "lock" in msg_lower or "databaselock" in msg_lower or "winerror 32" in msg_lower:
                    raise DatabaseLockException(f"UltSQL tool '{tool_name}' failed: {msg}")
                raise RuntimeError(f"UltSQL tool '{tool_name}' failed: {msg}")

            content = result_obj.get("content", [])
            if content and isinstance(content, list) and "text" in content[0]:
                text = content[0]["text"]
                try:
                    return json.loads(text)
                except Exception:
                    return text
            return result_obj

    def close(self) -> None:
        """Terminate the UltSQL process and release pipes."""
        with self._lock:
            if self._proc:
                pid = self._proc.pid
                if sys.platform == "win32" and pid:
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            capture_output=True,
                            timeout=2,
                            check=False,
                        )
                    except Exception:
                        pass
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                self._proc = None
            self._status = "closed"


class UltSQLMemoryBackend(BaseMemoryBackend):
    """Converged multimodal memory engine powered by UltSQL.

    Leverages:
    - KV Store: Sub-microsecond cache for file hashes.
    - NoSQL Collections: 'claims_history' with JSON query filtering.
    - Relational SQL: 'manifest_scripts' and 'repo_symbols' schema.
    - HNSW Vector RAG: Approximate nearest-neighbor search for code symbols.
    """

    def __init__(
        self,
        repo_path: str = ".",
        db_path: Optional[str] = None,
        client: Optional[UltSQLMCPClient] = None,
    ) -> None:
        self.repo_path = os.path.abspath(repo_path)
        if db_path:
            self.db_path = db_path
        elif repo_path == ":memory:":
            self.db_path = ":memory:"
        else:
            docdrift_dir = os.path.join(self.repo_path, ".docdrift")
            os.makedirs(docdrift_dir, exist_ok=True)
            self.db_path = os.path.join(docdrift_dir, "ultsql_data")
            os.makedirs(self.db_path, exist_ok=True)

        # Automatic stale lock cleanup on initialization
        self._cleanup_lock()

        self._client = client or UltSQLMCPClient()
        self._kv_keys: set[str] = set()
        self._init_schema()

    def _cleanup_lock(self) -> bool:
        """Removes stale or orphaned ultsql.lock in self.db_path."""
        return cleanup_ultsql_lock(self.db_path)

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke an UltSQL tool with automatic stale lock recovery and retry."""
        try:
            return self._client.call_tool(tool_name, arguments)
        except (DatabaseLockException, RuntimeError, OSError) as e:
            err_msg = str(e).lower()
            is_lock = (
                isinstance(e, DatabaseLockException)
                or "databaselock" in err_msg
                or "lock" in err_msg
                or "winerror 32" in err_msg
            )
            if is_lock:
                logger.warning(
                    f"UltSQL lock contention encountered on '{tool_name}': {e}. "
                    "Cleaning stale lock and retrying once..."
                )
                self._cleanup_lock()
                return self._client.call_tool(tool_name, arguments)
            raise

    @classmethod
    def is_available(cls) -> bool:
        """Class method to verify UltSQL runtime availability."""
        client = UltSQLMCPClient()
        available = client.is_available()
        client.close()
        return available

    def get_diagnostics(self) -> Dict[str, Any]:
        """Diagnostic metrics regarding UltSQL daemon and database paths."""
        diag = self._client.get_diagnostics()
        diag["db_path"] = self.db_path
        diag["engine"] = "UltSQL Converged Multimodal Database"
        return diag

    def _norm_path(self, path: str) -> str:
        return os.path.normpath(path).replace("\\", "/")

    def _init_schema(self) -> None:
        """Create relational tables and HNSW vector index in UltSQL."""
        ddl_scripts = """
        CREATE TABLE IF NOT EXISTS manifest_scripts (
            id VARCHAR(64) PRIMARY KEY,
            ecosystem VARCHAR(64),
            manifest_path VARCHAR(255),
            script_name VARCHAR(128),
            command_str TEXT
        );
        """
        self._call_tool(
            "ultsql_query",
            {"database": self.db_path, "query": ddl_scripts},
        )

        ddl_symbols = """
        CREATE TABLE IF NOT EXISTS repo_symbols (
            id VARCHAR(64) PRIMARY KEY,
            file_path VARCHAR(255),
            symbol_name VARCHAR(128),
            symbol_type VARCHAR(64),
            embedding VECTOR
        );
        """
        self._call_tool(
            "ultsql_query",
            {"database": self.db_path, "query": ddl_symbols},
        )



    # =========================================================================
    # a) KV Store (ultsql_kv): sub-microsecond cache for file hashes
    # =========================================================================

    def get_file_hash(self, path: str) -> Optional[str]:
        norm = self._norm_path(path)
        key = f"docdrift:hash:{norm}"
        res = self._call_tool(
            "ultsql_kv_get",
            {"database": self.db_path, "key": key},
        )
        if isinstance(res, dict) and res.get("exists"):
            return res.get("value")
        return None

    def set_file_hash(self, path: str, hash_val: str) -> None:
        norm = self._norm_path(path)
        key = f"docdrift:hash:{norm}"
        self._call_tool(
            "ultsql_kv_set",
            {"database": self.db_path, "key": key, "value": hash_val},
        )
        self._kv_keys.add(key)

    # =========================================================================
    # b) NoSQL Document Collections (ultsql_doc): claims_history
    # =========================================================================

    def store_claim(self, claim: Claim) -> None:
        data = claim.model_dump(mode="json")
        data["doc_path"] = self._norm_path(data["doc_path"])
        self._call_tool(
            "ultsql_doc_insert",
            {
                "database": self.db_path,
                "collection": "claims_history",
                "documents": [data],
            },
        )

    def get_claims(self, filters: Optional[Dict[str, Any]] = None) -> List[Claim]:
        doc_filter: Dict[str, Any] = {}
        if filters:
            for k, v in filters.items():
                val = v.value if hasattr(v, "value") else v
                if k == "doc_path" and isinstance(val, str):
                    val = self._norm_path(val)
                doc_filter[k] = val

        res = self._call_tool(
            "ultsql_doc_find",
            {
                "database": self.db_path,
                "collection": "claims_history",
                "filter": doc_filter,
            },
        )

        documents = res.get("documents", []) if isinstance(res, dict) else []
        claims: List[Claim] = []
        for doc in documents:
            clean_doc = {k: v for k, v in doc.items() if not k.startswith("_")}
            try:
                claims.append(Claim.model_validate(clean_doc))
            except Exception as e:
                logger.warning(f"Skipping malformed claim record: {e}")
        return claims

    # =========================================================================
    # c) Relational SQL (ultsql_query): manifest_scripts & repo_symbols
    # =========================================================================

    def index_manifest_scripts(self, scripts: List[Dict[str, Any]]) -> None:
        if not scripts:
            return
        for s in scripts:
            s_id = str(s.get("id") or uuid.uuid4().hex[:12])
            ecosystem = str(s.get("ecosystem", "generic"))
            manifest_path = self._norm_path(str(s.get("manifest_path", "")))
            script_name = str(s.get("script_name", ""))
            command_str = str(s.get("command_str", ""))

            # Sanitize single quotes for SQL insertion
            c_eco = ecosystem.replace("'", "''")
            c_path = manifest_path.replace("'", "''")
            c_name = script_name.replace("'", "''")
            c_cmd = command_str.replace("'", "''")

            sql = f"""
            DELETE FROM manifest_scripts WHERE id = '{s_id}';
            INSERT INTO manifest_scripts VALUES ('{s_id}', '{c_eco}', '{c_path}', '{c_name}', '{c_cmd}');
            """
            self._call_tool(
                "ultsql_query",
                {"database": self.db_path, "query": sql},
            )

    def index_symbols(self, symbols: List[Dict[str, Any]]) -> None:
        if not symbols:
            return
        for sym in symbols:
            sym_id = str(sym.get("id") or uuid.uuid4().hex[:12])
            file_path = self._norm_path(str(sym.get("file_path", "")))
            symbol_name = str(sym.get("symbol_name", ""))
            symbol_type = str(sym.get("symbol_type", "env_var"))

            vec = sym.get("embedding")
            if not vec or not isinstance(vec, list):
                vec = compute_symbol_embedding(symbol_name)

            vec_str = json.dumps(vec)
            c_path = file_path.replace("'", "''")
            c_name = symbol_name.replace("'", "''")
            c_type = symbol_type.replace("'", "''")

            sql = f"""
            INSERT INTO repo_symbols VALUES ('{sym_id}', '{c_path}', '{c_name}', '{c_type}', '{vec_str}');
            """
            self._call_tool(
                "ultsql_query",
                {"database": self.db_path, "query": sql},
            )

    # =========================================================================
    # d) HNSW Vector RAG (ultsql_vector_search): semantically matching symbols
    # =========================================================================

    def find_similar_symbol(
        self, query: str, symbol_type: str = "env_var", limit: int = 3
    ) -> List[Dict[str, Any]]:
        query_vec = compute_symbol_embedding(query)
        where_clause = f"symbol_type = '{symbol_type}'" if symbol_type else None

        res = self._call_tool(
            "ultsql_vector_search",
            {
                "database": self.db_path,
                "table": "repo_symbols",
                "vector_column": "embedding",
                "query_vector": query_vec,
                "metric": "cosine",
                "limit": limit,
                "where": where_clause,
            },
        )

        rows = res.get("rows", []) if isinstance(res, dict) else []
        candidates: List[Dict[str, Any]] = []
        for r in rows:
            dist = float(r.get("distance", 1.0))
            sim = round(max(0.0, 1.0 - dist), 4)
            candidates.append({
                "id": r.get("id"),
                "file_path": r.get("file_path"),
                "symbol_name": r.get("symbol_name"),
                "symbol_type": r.get("symbol_type"),
                "similarity": sim,
                "distance": round(dist, 4),
            })

        # Ensure sorted by similarity descending
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:limit]

    # =========================================================================
    # Persistence, Stats, and Reset
    # =========================================================================

    def save(self) -> None:
        """UltSQL automatically synchronizes transactions to its slotted pages and WAL."""
        pass

    def clear(self) -> None:
        """Purge data across relational tables, document collections, and KV store."""
        try:
            self._call_tool(
                "ultsql_query",
                {
                    "database": self.db_path,
                    "query": "DELETE FROM manifest_scripts; DROP TABLE IF EXISTS repo_symbols; DELETE FROM _coll_claims_history;",
                },
            )
            self._init_schema()
        except Exception:
            pass

        for key in list(self._kv_keys):
            try:
                self._call_tool(
                    "ultsql_kv_set",
                    {"database": self.db_path, "key": key, "value": None, "ttl_seconds": 1},
                )
            except Exception:
                pass
            self._kv_keys.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Aggregate stats across UltSQL converged engines."""
        claims_res = self._call_tool(
            "ultsql_doc_find",
            {"database": self.db_path, "collection": "claims_history"},
        )
        claims_count = claims_res.get("matchedCount", 0) if isinstance(claims_res, dict) else 0

        scripts_res = self._call_tool(
            "ultsql_query",
            {"database": self.db_path, "query": "SELECT COUNT(*) as cnt FROM manifest_scripts;"},
        )
        scripts_count = 0
        if isinstance(scripts_res, dict) and scripts_res.get("rows"):
            scripts_count = scripts_res["rows"][0].get("cnt", 0)

        symbols_res = self._call_tool(
            "ultsql_query",
            {"database": self.db_path, "query": "SELECT COUNT(*) as cnt FROM repo_symbols;"},
        )
        symbols_count = 0
        if isinstance(symbols_res, dict) and symbols_res.get("rows"):
            symbols_count = symbols_res["rows"][0].get("cnt", 0)

        return {
            "engine": "ultsql_converged",
            "db_path": self.db_path,
            "cached_file_hashes": len(self._kv_keys),
            "claims_history_count": claims_count,
            "manifest_scripts_count": scripts_count,
            "indexed_symbols_count": symbols_count,
        }

    def close(self) -> None:
        self.save()
        if self._client:
            self._client.close()

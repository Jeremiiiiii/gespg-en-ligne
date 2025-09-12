from contextlib import contextmanager
import os, sqlite3
from threading import RLock

DEFAULT_DB = "bom.sqlite3"
DB_PATH = os.environ.get("DB_PATH", DEFAULT_DB)

_db_lock = RLock()  

def _base_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(
        db_path,
        timeout=30,
        isolation_level="DEFERRED",   
        check_same_thread=True        
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn

@contextmanager
def ensure_conn(existing=None, db_path: str = DB_PATH):
    """
    Toujours préférer existing=None en Streamlit pour éviter le cross-thread.
    """
    if existing is not None:
        yield existing
    else:
        conn = _base_connect(db_path)
        try:
            yield conn
        finally:
            conn.close()

def execute(sql: str, params: tuple = ()):
    with ensure_conn() as conn:
        # Optionnel : sérialiser les écritures
        if sql.lstrip().upper().startswith(("INSERT","UPDATE","DELETE","REPLACE","CREATE","DROP","ALTER")):
            with _db_lock:
                return conn.execute(sql, params)
        else:
            return conn.execute(sql, params)

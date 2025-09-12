from contextlib import contextmanager
import os, sqlite3

DEFAULT_DB = "bom.sqlite3"
DB_PATH = os.environ.get("DB_PATH", DEFAULT_DB)

def _base_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(
        db_path,
        timeout=30,           # évite les 'database is locked' trop rapides
        isolation_level=None, # autocommit OFF -> géré par 'with conn:' (transactions explicites)
        check_same_thread=True  # PLUS SÛR : une connexion par thread
    )
    conn.row_factory = sqlite3.Row
    # PRAGMA cohérents partout
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn

def get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    return _base_connect(db_path)

@contextmanager
def ensure_conn(existing=None, db_path: str = DB_PATH):
    """Yields a connection. Si 'existing' est fourni, on le réutilise sans le fermer."""
    if existing is not None:
        yield existing
    else:
        with get_conn(db_path) as c:
            yield c

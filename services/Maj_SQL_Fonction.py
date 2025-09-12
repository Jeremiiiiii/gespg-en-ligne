import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

# ====== SQL helpers & services ======
from services.donnees import (
    _reload_tables_from_sql
    )

from services.interaction_DB import (
    delete_product
    )

from SQL.sql_bom import (
    get_conn,
    init_schema,
    upsert_product_row,   
    )

from SQL.db import DB_PATH

PF_COLUMNS = [
    "Index","Référence","Libellé produit","Composition","Couleur","Marque",
    "Famille","Libellé famille","Prix d'achat","PR","Unité","PV TTC",
    "Code liaison externe","Commentaire"
]

IMPORT_COL_MAP = {
    "Référence": ["refco", "référence", "reference", "ref", "ref_co", "REF_CO"],
    "Libellé produit": ["libelléproduit", "libelleproduit", "designation", "désignation", "LIBELLE PRODUIT"],
    "Composition": ["compo", "composition", "COMPO"],
    "Couleur": ["nomcouleur", "couleur", "COULEUR", "NOM COULEUR"],
    "Famille": ["codefamille", "famille", "CODE FAMILLE"],
    "Libellé famille": ["libellefamille", "libelléfamille", "LIBELLE FAMILLE"],
    "PR": ["pr", "PR"],
    "PV TTC": ["pvttc", "pv_ttc", "PV TTC"],
    "Unité": ["unite", "unité"],
    "Code liaison externe": ["segmentation", "code_liaison_externe", "SEGMENTATION"],
    "Prix d'achat": ["pa", "prixachat", "prix d'achat"],
    # "Commentaire": zone texte gérée par UI commentaires
}

# ─────────────────────────────────────────────────────────────
#                          UTILITAIRES
# ─────────────────────────────────────────────────────────────

def _clean_header(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lstrip("\ufeff").replace("ï»¿", "").strip().casefold()
    return re.sub(r"[\s_]+", "", s)

def _src_norm_lookup(df_src: pd.DataFrame) -> Dict[str, str]:
    return {_clean_header(c): c for c in df_src.columns}

def _detect_file_type(uploaded) -> str:
    name = getattr(uploaded, "name", "") or ""
    ext = Path(name).suffix.lower()
    if ext in (".xls", ".xlsx", ".xlsm"):
        return ext.lstrip(".")
    return "csv"

def _read_any_spreadsheet(uploaded, sheet: Optional[str] = None) -> pd.DataFrame:
    ftype = _detect_file_type(uploaded)
    uploaded.seek(0)
    if ftype in ("xlsx", "xlsm", "xls"):
        try:
            xls = pd.ExcelFile(uploaded, engine="openpyxl")
            target = sheet if (sheet and sheet in xls.sheet_names) else xls.sheet_names[0]
            return pd.read_excel(xls, sheet_name=target, dtype=str)
        except Exception:
            uploaded.seek(0)
            engine = "xlrd" if ftype=="xls" else "openpyxl"
            return pd.read_excel(uploaded, sheet_name=0, dtype=str, engine=engine)
    for sep in (None, ",", ";", "\t", "|"):
        try:
            uploaded.seek(0)
            return pd.read_csv(uploaded, sep=sep, dtype=str, engine="python")
        except Exception:
            continue
    raise ValueError("Impossible de lire le fichier fourni (CSV/XLS/XLSX).")


# -------- Console triée  --------
CONSOLE_KEY = "maj_console"
LOG_SEEN_KEY = "maj_log_seen"  # mémorise la dernière signature par catégorie

if CONSOLE_KEY not in st.session_state:
    st.session_state[CONSOLE_KEY] = []
if LOG_SEEN_KEY not in st.session_state:
    st.session_state[LOG_SEEN_KEY] = {}

def _append_console(line: str):
    t = datetime.now().strftime("[%H:%M:%S]")
    st.session_state[CONSOLE_KEY].append(f"{t} {line}")

def _log_dedup(cat: str, sig: tuple, line: str):
    """
    cat = catégorie logique ("import", "clean", "base", "mapping", "indexes", "audit"...)
    sig = signature immuable de l'info (tuple d’éléments stables). Si sig identique => on ne log pas.
    """
    seen = st.session_state[LOG_SEEN_KEY]
    if seen.get(cat) == sig:
        return  # rien de nouveau => pas de bruit
    seen[cat] = sig
    _append_console(line)

# ====== DB helpers ======
def _get_db_path() -> str:
    return st.session_state.get("DB_PATH", DB_PATH)

def _detect_index_column_name(conn) -> str:
    cur = conn.execute("PRAGMA table_info(products);")
    cols = [r[1] for r in cur.fetchall()]
    for c in ("product_index","Index","index","idx"):
        if c in cols:
            return c
    return "product_index"

def _count_products() -> int:
    with get_conn(_get_db_path()) as conn:
        try:
            r = conn.execute("SELECT COUNT(1) FROM products;").fetchone()
            return int(r[0]) if r else 0
        except Exception:
            return 0

# ====== Prétraitement & Mapping ======
@st.cache_data(show_spinner=False)
def preprocess_source_df(df_src: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if df_src is None or df_src.empty:
        return df_src, {"non_empty_rows": 0, "dropped_duplicates": 0, "after_rows": 0, "dedup_on": "—"}

    df_str = df_src.astype(object).where(df_src.notna(), "").applymap(lambda x: str(x).strip())
    non_empty_mask = (df_str != "").any(axis=1)
    df_nonempty = df_src.loc[non_empty_mask].copy()
    non_empty_count = int(non_empty_mask.sum())

    norm2src = _src_norm_lookup(df_nonempty)
    dedup_on = norm2src.get("refco")
    if dedup_on:
        before = len(df_nonempty)
        df_nonempty = df_nonempty.drop_duplicates(subset=[dedup_on], keep="first")
        dropped = before - len(df_nonempty)
    else:
        dropped = 0

    return df_nonempty.reset_index(drop=True), {
        "non_empty_rows": non_empty_count,
        "dropped_duplicates": int(dropped),
        "after_rows": int(len(df_nonempty)),
        "dedup_on": dedup_on or "REF_CO",
    }

@st.cache_data(show_spinner=False)
def map_source_to_pf(
    df_src: pd.DataFrame,
    default_brand: str = "",
    *,
    comment_mode: str = "Aucun",
    global_comment: str = "",
    comment_blocks: Optional[List[Dict[str, Any]]] = None
) -> pd.DataFrame:
    """Mapping -> DataFrame PF + gestion commentaires (aucun, global, spécifiques par blocs)."""
    if df_src is None or df_src.empty:
        return pd.DataFrame(columns=PF_COLUMNS)

    norm2src = _src_norm_lookup(df_src)
    out: Dict[str, pd.Series] = {}

    # Index
    idx_col = None
    for c in df_src.columns:
        if _clean_header(c) == "index":
            idx_col = c
            break
    out["Index"] = df_src[idx_col].astype(str) if idx_col else pd.Series([""] * len(df_src))

    # Colonnes mappées
    for dest, aliases in IMPORT_COL_MAP.items():
        src_name = None
        for a in aliases:
            src_name = norm2src.get(a)
            if src_name:
                break
        if src_name is not None:
            out[dest] = df_src[src_name].astype(str)

    # Marque par défaut
    if "Marque" not in out and default_brand:
        out["Marque"] = pd.Series([default_brand] * len(df_src), dtype=str)

    # Commentaires
    comments = pd.Series("", index=df_src.index)
    if comment_mode == "Global":
        comments[:] = global_comment or ""
    elif comment_mode.startswith("Spécifique"):
        lib_series = out.get("Libellé produit", pd.Series("", index=df_src.index)).astype(str)
        for block in (comment_blocks or []):
            labels = [str(x).strip() for x in (block.get("labels") or []) if str(x).strip() != ""]
            txt = (block.get("comment") or "").strip()
            if labels and txt:
                comments.loc[lib_series.isin(set(labels))] = txt
    out["Commentaire"] = comments

    # Prix d'achat forcé à 0 (aucun produit ajouté n'est acheté)
    out["Prix d'achat"] = pd.Series(["0"] * len(df_src), dtype=str)

    df = pd.DataFrame(out).fillna("")
    df["Index"] = df["Index"].astype(str).str.strip().str.upper()

    # S'assurer de l'ordre/présence des colonnes PF
    for c in PF_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    return df[PF_COLUMNS]

# ====== Index Auto : max+1 séquentiel (pas de réutilisation) ======
FIXED_PREFIX_FALLBACK = "PF"
FIXED_PAD_FALLBACK = 5

def _split_index(token: str) -> Optional[tuple[str,int,int]]:
    if token is None:
        return None
    m = re.match(r"^([A-Za-z]+)(\d+)$", str(token).strip())
    if not m:
        return None
    prefix, digits = m.group(1), m.group(2)
    try:
        return prefix.upper(), int(digits), len(digits)
    except Exception:
        return None

def _read_index_scheme(conn) -> tuple[str,int,int]:
    """Retourne (prefix, pad, max_num) détectés en base (favorise PF)."""
    idx_col = _detect_index_column_name(conn)
    try:
        rows = conn.execute(f"SELECT {idx_col} FROM products;").fetchall()
    except Exception:
        rows = []
    max_by_prefix: Dict[str, tuple[int,int]] = {}
    for (tok,) in rows:
        sp = _split_index(tok)
        if not sp:
            continue
        pfx, num, pad = sp
        cur = max_by_prefix.get(pfx)
        if cur is None or num > cur[0]:
            max_by_prefix[pfx] = (num, pad)
    if "PF" in max_by_prefix:
        n, p = max_by_prefix["PF"]
        return "PF", p, n
    if max_by_prefix:
        pfx = max(max_by_prefix.items(), key=lambda kv: kv[1][0])[0]
        n, p = max_by_prefix[pfx]
        return pfx, p, n
    return FIXED_PREFIX_FALLBACK, FIXED_PAD_FALLBACK, 0

def assign_indexes_sequential(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str,Any]]:
    if df is None or df.empty:
        return df, {"assigned": 0}
    with get_conn(_get_db_path()) as conn:
        init_schema(conn)
        prefix, pad, max_num = _read_index_scheme(conn)
    need_mask = ~df["Index"].astype(str).map(lambda s: _split_index(s) is not None)
    need_count = int(need_mask.sum())
    if need_count == 0:
        return df, {"assigned": 0, "prefix": prefix, "pad": pad}
    start = max_num + 1
    seq = [start + i for i in range(need_count)]
    df.loc[need_mask, "Index"] = [f"{prefix}{n:0{pad}d}" for n in seq]
    return df, {
        "assigned": need_count,
        "prefix": prefix,
        "pad": pad,
        "first": f"{prefix}{seq[0]:0{pad}d}",
        "last":  f"{prefix}{seq[-1]:0{pad}d}",
    }

# ─────────────────────────────────────────────────────────────
#                          AUDIT/UNDO
# ─────────────────────────────────────────────────────────────

def _ts_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _fetch_audit_since(ts_from: str) -> pd.DataFrame:
    conn = get_conn(_get_db_path())
    init_schema(conn)
    q = (
        "SELECT id, ts, user, action, table_name, rec_key, field, old_value, new_value, before_json, after_json "
        "FROM audit_log WHERE ts >= ? AND table_name='products' "
        "AND action IN ('CREATE_ROW','UPDATE_FIELD','UPDATE_ROW') ORDER BY ts ASC, id ASC"
    )
    return pd.read_sql_query(q, conn, params=[ts_from])

def _group_changes(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Regroupe par produit (rec_key=Index). [{index, label, created, fields, ids}]"""
    if df is None or df.empty:
        return []
    groups: Dict[str, Dict[str, Any]] = {}
    for r in df.to_dict(orient="records"):
        if r.get("table_name") == "bom_edges":
            continue
        ix = str(r.get("rec_key") or "—")
        g = groups.setdefault(ix, {"index": ix, "label": "—", "created": False, "fields": {}, "ids": []})
        g["ids"].append(int(r["id"]))
        if r.get("action") == "CREATE_ROW":
            g["created"] = True
            try:
                after = json.loads(r.get("after_json") or "{}")
                g["label"] = after.get("libelle_produit") or g["label"]
            except Exception:
                pass
        elif r.get("action") == "UPDATE_FIELD":
            field = r.get("field")
            if field:
                g["fields"].setdefault(field, {"old": r.get("old_value"), "new": r.get("new_value")})
        elif r.get("action") == "UPDATE_ROW" and g.get("label") == "—":
            try:
                after = json.loads(r.get("after_json") or "{}")
                g["label"] = after.get("libelle_produit") or g["label"]
            except Exception:
                pass
    return list(groups.values())

def _undo_created(index: str, audit_ids: List[int]) -> tuple[bool, str]:
    """Supprime le produit créé et purge ses logs."""
    with get_conn(_get_db_path()) as conn:
        ok, msg = delete_product(conn, index, user=st.session_state.get("current_user"))
        if not ok:
            return False, msg
        if audit_ids:
            conn.execute(f"DELETE FROM audit_log WHERE id IN ({','.join(['?']*len(audit_ids))});", audit_ids)
        conn.commit()
    _reload_tables_from_sql()
    return True, "Création annulée."

def _undo_updates(index: str, fields_old: Dict[str, Any], audit_ids: List[int]) -> tuple[bool, str]:
    """Restaure les champs modifiés via UPDATE_FIELD en une passe sans reloguer."""
    payload: Dict[str, Any] = {"product_index": index}
    for f, pair in (fields_old or {}).items():
        payload[f] = pair.get("old")
    try:
        with get_conn(_get_db_path()) as conn:
            upsert_product_row(conn, payload, user=None, audit=False)
            if audit_ids:
                conn.execute(f"DELETE FROM audit_log WHERE id IN ({','.join(['?']*len(audit_ids))});", audit_ids)
            conn.commit()
    except Exception as e:
        return False, f"Échec de l'annulation: {type(e).__name__}: {e}"
    _reload_tables_from_sql()
    return True, "Modification annulée."

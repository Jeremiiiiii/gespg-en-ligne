# ====== Gardes & imports ======
import os
import sqlite3
import unicodedata
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Iterable, Optional, Set
from services.interaction_DB import delete_product  

import pandas as pd
import streamlit as st

# ============================== DB ============================================
DEFAULT_DB = "bom.sqlite3"
DB_PATH = st.session_state.get("DB_PATH") or os.environ.get("DB_PATH") or DEFAULT_DB

from SQL.db import get_conn

# ============================== CHAMPS & POIDS ================================
TEXT_FIELDS = [
    "reference","libelle_produit","composition","couleur","marque",
    "famille","libelle_famille","unite","code_liaison_externe",
    "fournisseur","designation_fournisseur"
]
NUM_FIELDS = ["prix_achat","pr","pv_ttc"]
ALL_FIELDS = TEXT_FIELDS + NUM_FIELDS  # (hors 'product_index/kind')

DEFAULT_WEIGHTS: Dict[str, float] = {
    "reference": 0.25,
    "libelle_produit": 0.25,
    "composition": 0.04,
    "couleur": 0.25,
    "marque": 0.04,
    "famille": 0.03,
    "libelle_famille": 0.03,
    "unite": 0.02,
    "code_liaison_externe": 0.02,
    "fournisseur": 0.02,
    "designation_fournisseur": 0.01,
    "prix_achat": 0.03,
    "pr": 0.01,
    "pv_ttc": 0.01,
}

def _ensure_weights_sum_to_one(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(w.get(k, 0.0) for k in ALL_FIELDS)
    if s <= 0: return DEFAULT_WEIGHTS.copy()
    return {k: w.get(k, 0.0) / s for k in ALL_FIELDS}

# ============================== LOAD PRODUITS =================================
@st.cache_data(show_spinner=False)
def load_products() -> pd.DataFrame:
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT rowid, * FROM products;", conn)
    for col in NUM_FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ============================== UTILS =========================================
def norm_txt(x: Any) -> str:
    if x is None: return ""
    s = str(x)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s, flags=re.I).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def jaccard(a: str, b: str) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    A, B = set(a.split()), set(b.split())
    if not A and not B: return 1.0
    inter, union = len(A & B), len(A | B)
    return inter / union if union else 0.0

def eq_text(a: Any, b: Any) -> float:
    na, nb = norm_txt(a), norm_txt(b)
    if na == nb and na != "": return 1.0
    return jaccard(na, nb)

def eq_num(a: Any, b: Any, rel_tol: float = 0.01, abs_tol: float = 0.01) -> float:
    try:
        fa = float(a) if a is not None else None
        fb = float(b) if b is not None else None
    except Exception:
        return 0.0
    if fa is None and fb is None: return 1.0
    if fa is None or fb is None: return 0.0
    if abs(fa - fb) <= max(abs_tol, rel_tol * max(abs(fa), abs(fb))): return 1.0
    return 0.0

def all_equal(row_a: pd.Series, row_b: pd.Series) -> bool:
    for c in ALL_FIELDS:
        if c in NUM_FIELDS:
            if eq_num(row_a.get(c), row_b.get(c)) < 1.0: return False
        else:
            if eq_text(row_a.get(c), row_b.get(c)) < 1.0: return False
    return True

def similarity(row_a: pd.Series, row_b: pd.Series, weights: Dict[str, float]) -> float:
    if all_equal(row_a, row_b): return 1.0
    score = 0.0
    for c in ALL_FIELDS:
        w = weights.get(c, 0.0)
        if w <= 0: continue
        sim = eq_num(row_a.get(c), row_b.get(c)) if c in NUM_FIELDS else eq_text(row_a.get(c), row_b.get(c))
        score += w * sim
    return min(max(score, 0.0), 1.0)

# --------- Paires au sein d’un même type (PF↔PF ou SF↔SF) ----------
def build_keys(df: pd.DataFrame) -> Dict[str, Dict[str, List[int]]]:
    maps: Dict[str, Dict[str, List[int]]] = {"reference": defaultdict(list), "libelle": defaultdict(list), "code": defaultdict(list)}
    if "reference" in df.columns:
        for i, v in enumerate(df["reference"].tolist()):
            k = norm_txt(v)
            if k: maps["reference"][k].append(i)
    if "libelle_produit" in df.columns:
        for i, v in enumerate(df["libelle_produit"].tolist()):
            k = norm_txt(v)
            if k: maps["libelle"][k].append(i)
            short = " ".join(k.split()[:4])
            if short: maps["libelle"][short].append(i)
    if "code_liaison_externe" in df.columns:
        for i, v in enumerate(df["code_liaison_externe"].tolist()):
            k = norm_txt(v)
            if k: maps["code"][k].append(i)
    return maps

def candidate_pairs_within(df: pd.DataFrame) -> Iterable[Tuple[int, int, str]]:
    keys = build_keys(df)
    seen: Set[Tuple[int, int]] = set()

    def emit(bucket: List[int], reason: str):
        bucket = sorted(set(bucket))
        for a in range(len(bucket)):
            for b in range(a + 1, len(bucket)):
                i, j = bucket[a], bucket[b]
                if (i, j) not in seen:
                    seen.add((i, j))
                    yield (i, j, reason)

    for _k, bucket in keys["reference"].items():
        for t in emit(bucket, "reference"): yield t
    for _k, bucket in keys["libelle"].items():
        for t in emit(bucket, "libelle"): yield t
    for _k, bucket in keys["code"].items():
        for t in emit(bucket, "code"): yield t

@dataclass
class EdgeH:
    i: int
    j: int
    score: float
    tag: str  # "PF" ou "SF"

def build_graph_and_clusters_homogeneous(df: pd.DataFrame, weights: Dict[str, float], threshold: float, tag: str) -> Tuple[List[EdgeH], List[List[Tuple[str,int]]]]:
    edges: List[EdgeH] = []
    parent: Dict[Tuple[str,int], Tuple[str,int]] = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    for i, j, _r in candidate_pairs_within(df):
        s = similarity(df.iloc[i], df.iloc[j], weights)
        if s >= threshold:
            edges.append(EdgeH(i, j, s, tag))
            union((tag, i), (tag, j))

    groups: Dict[Tuple[str,int], List[Tuple[str,int]]] = defaultdict(list)
    for e in edges:
        groups[find((tag, e.i))].append((tag, e.i))
        groups[find((tag, e.j))].append((tag, e.j))

    lots: List[List[Tuple[str,int]]] = []
    for _root, members in groups.items():
        uniq = list(dict.fromkeys(members))
        if len(uniq) >= 2:
            lots.append(uniq)
    return edges, lots

# ---------- Sérialisation des lots (stable dans la session) ----------
def signature_from_members(members: List[Dict[str, Any]]) -> Tuple[str, ...]:
    cols = []
    for c in ALL_FIELDS:
        vals = []
        for m in members:
            v = m.get(c)
            vals.append(norm_txt(v) if c not in NUM_FIELDS else v)
        if pd.Series(vals, dtype="object").nunique(dropna=False) > 1:
            cols.append(c)
    return tuple(sorted(cols))

def serialize_lots(lots: List[List[Tuple[str,int]]], df_pf: pd.DataFrame, df_sf: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    next_id = 0
    for lot in lots:
        members: List[Dict[str, Any]] = []
        tag = None
        for t, idx in lot:
            row = (df_pf if t == "PF" else df_sf).iloc[idx]
            rec = {
                "tag": t,
                "product_index": row["product_index"],
                "rowid": int(row["rowid"]),
            }
            for f in ALL_FIELDS:
                rec[f] = row.get(f)
            members.append(rec)
            tag = t
        keep_index = min(members, key=lambda m: m["rowid"])["product_index"]
        sig = signature_from_members(members)
        out.append({"id": next_id, "tag": tag, "members": members, "keep_index": keep_index, "signature": sig, "resolved": False})
        next_id += 1
    return out

# ============================== AUDIT & SUPPRESSION ===========================
def invalidate_data_cache():
    try:
        # Efface UNIQUEMENT le cache de load_products()
        load_products.clear()
    except Exception:
        # Fallback (certaines versions) : efface tous les caches data de la session
        st.cache_data.clear()

def fetch_product_as_json(conn: sqlite3.Connection, product_index: str) -> str:
    cur = conn.execute("SELECT * FROM products WHERE product_index=?;", (product_index,))
    row = cur.fetchone()
    if not row: return ""
    d = {k: row[k] for k in row.keys()}
    return pd.Series(d).to_json(force_ascii=False)

def audit_log(conn: sqlite3.Connection, user: Optional[str], action: str, table_name: str, rec_key: str, before_json: Optional[str], after_json: Optional[str], note: str=""):
    conn.execute(
        'INSERT INTO audit_log(ts,user,action,table_name,rec_key,before_json,after_json,note) VALUES(datetime("now"),?,?,?,?,?,?,?)',
        (user, action, table_name, rec_key, before_json, after_json, note)
    )

def _cascade_delete(conn: sqlite3.Connection, idx: str, user: Optional[str], lot_id: int, keep: str) -> int:
    before = fetch_product_as_json(conn, idx)
    cur1 = conn.execute("DELETE FROM bom_edges WHERE parent_index=? OR child_index=?;", (idx, idx))
    deleted_edges = cur1.rowcount or 0
    cur2 = conn.execute("DELETE FROM products WHERE product_index=?;", (idx,))
    audit_log(
        conn=conn, user=user, action="ELIMINATION_DOUBLONS",
        table_name="products", rec_key=idx, before_json=before, after_json=None,
        note=f"Conserver={keep} ; Cascade bom_edges={deleted_edges} ; Lot={lot_id}"
    )
    return cur2.rowcount or 0

def delete_duplicates_for_lot(lot: Dict[str, Any], user: Optional[str]) -> int:
    """Supprime tous les membres d’un lot sauf le plus ancien. Retourne nb suppressions."""
    keep = lot["keep_index"]
    to_delete = [m["product_index"] for m in lot["members"] if m["product_index"] != keep]
    if not to_delete:
        lot["resolved"] = True
        lot["members"] = [m for m in lot["members"] if m["product_index"] == keep]
        return 0

    total = 0
    with get_conn() as conn:
        for idx in to_delete:
            ok = False
            if delete_product:
                try:
                    ok, msg = delete_product(conn, idx, user=user)
                except Exception as e:
                    ok, msg = False, f"Exception delete_product: {e}"
            if not delete_product or not ok:
                total += _cascade_delete(conn, idx, user=user, lot_id=lot['id'], keep=keep)
            else:
                try:
                    before = ""  # inconnu après delete ; ce log reste indicatif
                    audit_log(conn, user, "ELIMINATION_DOUBLONS", "products", idx, before, None, f"Conserver={keep} ; Lot={lot['id']} (via delete_product)")
                except Exception:
                    pass
        conn.commit()
    invalidate_data_cache()

    lot["members"] = [m for m in lot["members"] if m["product_index"] == keep]
    lot["resolved"] = True
    return total


# ============================== AUTH UTILISATEUR ===============================

def get_logged_user() -> str:
    ss = st.session_state
    for k in ("current_user", "user", "username", "login", "email"):
        v = ss.get(k)
        if isinstance(v, dict):
            for kk in ("login", "username", "email", "name"):
                if v.get(kk): return str(v[kk])
        if v: return str(v)
    return os.environ.get("USER") or os.environ.get("USERNAME") or "—"
#donnees

from typing import Dict,Any,List
import pandas as pd
import streamlit as st

# ───────────────── Base SQLite ─────────────────

from SQL.sql_bom import (
    init_schema,
)
from SQL.db import get_conn,DB_PATH

class Kind:
    PF = "PF"  # produits finis
    SF = "SF"  # produits semi-finis


PF_COLUMNS = [
    "Index","Référence","Libellé produit","Composition","Couleur","Marque",
    "Famille","Libellé famille","Prix d'achat","PR","Unité","PV TTC",
    "Code liaison externe","Commentaire"
]
SF_COLUMNS = [
    "Index","Libellé produit","Composition","Couleur","Unité","Fournisseur",
    "Désignation fournisseur","Prix d'achat","PR","Code liaison externe",
    "Commentaire","Composé","Marque","Famille","Référence","PV TTC","Libellé famille"
]

# ───────────────── Normalisation d’Index  ─────────────────

def _normalize_index_value(v):
    """Normalise une valeur d'Index en string propre (gère NaN, '12.0'→'12', espaces)."""
    s = "" if pd.isna(v) else str(v).strip()
    if s == "":
        return s
    try:
        f = float(s.replace(",", "."))
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return s

def load_products(kind: str) -> pd.DataFrame:
    """Lit depuis SQL.products selon le type (PF/SF) et renvoie un DataFrame compatible UI."""
    db_path = st.session_state.get("DB_PATH", DB_PATH)
    conn = get_conn(db_path)
    init_schema(conn)

    if kind == Kind.PF:
        q = """
        SELECT
          product_index AS "Index",
          reference     AS "Référence",
          libelle_produit AS "Libellé produit",
          composition   AS "Composition",
          couleur       AS "Couleur",
          marque        AS "Marque",
          famille       AS "Famille",
          libelle_famille AS "Libellé famille",
          prix_achat    AS "Prix d'achat",
          pr            AS "PR",
          unite         AS "Unité",
          pv_ttc        AS "PV TTC",
          code_liaison_externe AS "Code liaison externe",
          commentaire   AS "Commentaire"
        FROM products
        WHERE kind='PF'
        ORDER BY product_index;
        """
        df = pd.read_sql_query(q, conn).fillna("")
        if "Index" in df.columns:
            df["Index"] = df["Index"].apply(_normalize_index_value)
        for c in PF_COLUMNS:
            if c not in df.columns: df[c] = ""
        return df[PF_COLUMNS]

    elif kind == Kind.SF:
        q = """
        SELECT
          product_index AS "Index",
          libelle_produit AS "Libellé produit",
          composition   AS "Composition",
          couleur       AS "Couleur",
          unite         AS "Unité",
          fournisseur   AS "Fournisseur",
          designation_fournisseur AS "Désignation fournisseur",
          prix_achat    AS "Prix d'achat",
          pr            AS "PR",
          code_liaison_externe AS "Code liaison externe",
          commentaire   AS "Commentaire",
          CASE WHEN compose IS NULL THEN '' ELSE CAST(compose AS TEXT) END AS "Composé",
          marque        AS "Marque",
          famille       AS "Famille",
          reference     AS "Référence",
          pv_ttc        AS "PV TTC",
          libelle_famille AS "Libellé famille"
        FROM products
        WHERE kind='SF'
        ORDER BY product_index;
        """
        df = pd.read_sql_query(q, conn).fillna("")
        if "Index" in df.columns:
            df["Index"] = df["Index"].apply(_normalize_index_value)
        for c in SF_COLUMNS:
            if c not in df.columns: df[c] = ""
        return df[SF_COLUMNS]

    else:
        return pd.DataFrame()

def normalize_selected_list(sel: Any) -> List[Dict]:
    if sel is None:
        return []
    if isinstance(sel, pd.DataFrame):
        return [] if sel.empty else sel.to_dict(orient='records')
    if isinstance(sel, list):
        return sel
    return []

def filtrer_df_par_categories(df: pd.DataFrame, categories: List[str]) -> pd.DataFrame:
    colonnes_existe = [col for col in categories if col in df.columns]
    return df[colonnes_existe]


# ───────────────── Outils DataFrame en session  ─────────────────

def replace_session_df_in_place(session_key: str, new_df: pd.DataFrame):
    new_df = new_df.reset_index(drop=True).copy()
    if session_key not in st.session_state or not isinstance(st.session_state[session_key], pd.DataFrame):
        st.session_state[session_key] = new_df
        return
    old = st.session_state[session_key]
    to_drop = [c for c in old.columns if c not in new_df.columns]
    if to_drop:
        old.drop(columns=to_drop, inplace=True)
    for c in new_df.columns:
        if c not in old.columns:
            old[c] = ""
    old.drop(old.index, inplace=True)
    for c in new_df.columns:
        old.loc[:, c] = new_df[c].values
    old.reset_index(drop=True, inplace=True)

def build_index_map_normalized(df: pd.DataFrame, index_col: str = "Index") -> Dict[str, int]:
    if df is None or index_col not in df.columns:
        return {}
    ser = df[index_col].astype(str).map(_normalize_index_value).fillna("").astype(str)
    out: Dict[str, int] = {}
    for i, v in enumerate(ser.values):
        if v and v not in out:
            out[v] = i
    return out

def ensure_index_map_in_state(state_key: str):
    df = st.session_state.get(state_key)
    if isinstance(df, pd.DataFrame):
        st.session_state[f"{state_key}__index_map"] = build_index_map_normalized(df)
    else:
        st.session_state[f"{state_key}__index_map"] = {}

def update_index_map_for_state(state_key: str, index_col: str = "Index"):
    df = st.session_state.get(state_key)
    if isinstance(df, pd.DataFrame) and index_col in df.columns:
        st.session_state[f"{state_key}__index_map"] = build_index_map_normalized(df, index_col)
    else:
        st.session_state[f"{state_key}__index_map"] = {}

def _bump_aggrid_refresh():
    st.session_state["aggrid_refresh"] = st.session_state.get("aggrid_refresh", 0) + 1


def _reload_tables_from_sql() -> None:
    try:
        st.cache_data.clear()
    except Exception:
        pass

    df_pf = load_products(Kind.PF)
    df_sf = load_products(Kind.SF)

    st.session_state.df_data  = df_pf.copy()
    st.session_state.df_data2 = df_sf.copy()

    st.session_state["df_full"]  = st.session_state.df_data.copy()
    st.session_state["df2_full"] = st.session_state.df_data2.copy()

    _bump_aggrid_refresh()  


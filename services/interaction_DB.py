#interaction_DB
import sqlite3
from typing import List, Dict, Optional, Any, Tuple
from typing import Dict
import pandas as pd
import streamlit as st

from services.donnees import (
    _normalize_index_value,
    _reload_tables_from_sql,
    DB_PATH,
    Kind
)

from SQL.sql_bom import (
    init_schema,
    add_edge, remove_edge,
    compute_bom_tree_sql,              
    get_conn,
    import_products_from_dataframes,
    log_audit,
    upsert_products_diff
)

_conn = get_conn(DB_PATH)

# ============================== ArbreParenté  ==============================

def get_children(conn: sqlite3.Connection, parent: str) -> List[Dict]:
    p = _normalize_index_value(parent)
    rows = conn.execute(
        "SELECT child_index, quantity FROM bom_edges WHERE parent_index=? ORDER BY child_index;",
        (p,)
    ).fetchall()
    return [{"Index": r["child_index"], "Quantité_pour_parent": float(r["quantity"])} for r in rows]


def get_children_info(tree: Dict, target_index: str) -> Dict[str, Dict[str, Any]]:
    def search(node: Dict) -> Optional[List[Dict]]:
        if node["Index"] == target_index:
            return node.get("children", [])
        for c in node.get("children", []):
            found = search(c)
            if found is not None:
                return found
        return None
    children = search(tree)
    if children is None:
        return {}
    return {
        c["Index"]: {
            "Quantité_pour_parent": c.get("Quantité_pour_parent"),
            "PR": c.get("PR")
        }
        for c in children
    }


def get_children_of_parent(parent_index: str, parent_state_key: str, component_state_key: Optional[str]) -> List[Dict]:
    """
    Version SQL : lit bom_edges, puis enrichit (Libellé, Unité) depuis le DF composant si dispo.
    """
    parent_idx = _normalize_index_value(parent_index)
    if not parent_idx:
        return []

    rows = get_children(_conn, parent_idx)  # [{'Index': child, 'Quantité_pour_parent': q}, ...]

    df_comp = st.session_state.get(component_state_key) if component_state_key else None
    if isinstance(df_comp, pd.DataFrame) and "Index" in df_comp.columns:
        idx_map = { _normalize_index_value(v): i for i, v in enumerate(df_comp["Index"].astype(str).values) }
        for r in rows:
            pos = idx_map.get(_normalize_index_value(r["Index"]))
            if pos is not None:
                row = df_comp.iloc[pos].to_dict()
                r.update({
                    "Libellé produit": row.get("Libellé produit", row.get("Désignation fournisseur","")),
                    "Unité": row.get("Unité", row.get("Unite",""))
                })
            else:
                r["missing"] = True
    return rows

def add_child_to_parent_v2(
    parent_state_key: str,
    component_state_key: Optional[str],
    parent_index: str, child_index: str, quantite: Optional[str] = None,
    rerun_after: bool = True, clear_cache_after_write: bool = True
) -> Dict[str, Any]:
    try:
        q = quantite if quantite is not None else "0"
        user = st.session_state.get("current_user")
        add_edge(_conn, parent_index, child_index, q, user=user)
        return {"ok": True, "msg": "Lien créé/mis à jour en SQL."}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def get_levels(tree: Dict) -> List[List[str]]:
    levels = []
    current_level = [tree]
    while current_level:
        indices = [node["Index"] for node in current_level]
        levels.append(indices)
        next_level = []
        for node in current_level:
            next_level.extend(node.get("children", []))
        current_level = next_level
    return levels

# ============================== Supression  ==============================

def delete_product(conn: sqlite3.Connection, index: str, user: Optional[str] = None) -> Tuple[bool, str]:
    ix = _normalize_index_value(index)

    # 1) Existence
    row = conn.execute(
        "SELECT * FROM products WHERE product_index=?;",
        (ix,)
    ).fetchone()
    if row is None:
        return False, f"Index introuvable: {ix}"

    # 2) Liens empêchant la suppression (RESTRICT)
    childs = [r["child_index"] for r in conn.execute(
        "SELECT child_index FROM bom_edges WHERE parent_index=?;",
        (ix,)
    ).fetchall()]
    parents = [r["parent_index"] for r in conn.execute(
        "SELECT parent_index FROM bom_edges WHERE child_index=?;",
        (ix,)
    ).fetchall()]

    if childs or parents:
        parts = []
        if childs:
            parts.append("a des enfants: " + ", ".join(childs))
        if parents:
            parts.append("est enfant de: " + ", ".join(parents))
        return False, "Pas supprimable car " + " et ".join(parts) + "."

    # 3) Suppression + audit en transaction
    try:
        with conn:
            conn.execute("DELETE FROM products WHERE product_index=?;", (ix,))
            # journalise l'état avant (after = None)
            log_audit(
                    conn, user,
                    action="DELETE_PRODUCT",
                    table="products",
                    rec_key=ix,
                    before_obj=dict(row),
                    after_obj=None,
                    note=None
                )

        return True, "Supprimé."
    except sqlite3.IntegrityError:
        # Garde-fou si d'autres FK existent ailleurs
        return False, "Suppression refusée (contrainte d'intégrité)."



def remove_child_from_parent_v2(
    parent_state_key: str,
    component_state_key: Optional[str],
    parent_index: str, child_index: str,
    rerun_after: bool = True, clear_cache_after_write: bool = True
) -> Dict[str, Any]:
    try:
        user = st.session_state.get("current_user")
        remove_edge(_conn, parent_index, child_index, user=user)
        return {"ok": True, "msg": "Lien supprimé en SQL."}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


def delete_rows_by_index(state_key: str, kind: Kind, indices_to_delete: List[str]) -> None:
    """
    Supprime des produits par Index **en SQL** (RESTRICT si liés),
    puis recharge **proprement** les DataFrames depuis SQL et force un rerun.
    """
    indices = [_normalize_index_value(ix) for ix in indices_to_delete if str(ix).strip()]
    indices = sorted(set(indices))
    if not indices:
        st.warning("Aucune ligne sélectionnée à supprimer.")
        return

    blocked, deleted = [], []
    for ix in indices:
        ok, msg = delete_product(_conn, ix)
        if ok:
            deleted.append(ix)
        else:
            blocked.append((ix, msg))

    if blocked:
        for ix, msg in blocked:
            st.error(f"{ix} : {msg}")

    if not deleted:
        st.info("Aucune suppression effectuée.")
        return

    try:
        st.cache_data.clear()
    except Exception:
        pass

    _reload_tables_from_sql()                  
    st.session_state["ag_main_selected"] = []  
    st.toast(f"{len(deleted)} produit(s) supprimé(s).", icon="🗑️")

    st.rerun()

# ============================== Sauvegarde  ==============================

def save_products(kind: Kind, df: pd.DataFrame) -> None:
    """
    Upsert vers SQL avec audit cellule par cellule :
    - utilise upsert_products_diff => log 'UPDATE_FIELD' (par champ) + 'CREATE_ROW'
    - ne logge pas tout le dataset, seulement ce qui change réellement
    """
    db_path = st.session_state.get("DB_PATH", DB_PATH)
    conn = get_conn(db_path)
    init_schema(conn)

    # Sécuriser df
    if df is None:
        df_use = pd.DataFrame()
    else:
        df_use = df.copy().fillna("")

    # L'utilisateur courant pour l'audit
    user = st.session_state.get("current_user")

    # Appel DIFF : PF dans paramètre df_pf, SF dans df_sf
    if kind == Kind.PF:
        upsert_products_diff(conn, df_pf=df_use, df_sf=None, user=user)
    else:
        upsert_products_diff(conn, df_pf=None, df_sf=df_use, user=user)



def calculer_prix_sql(root_index: str) -> Dict[str, Any]:
    return compute_bom_tree_sql(_conn, root_index)  

# Utilitaire local : récupère un PR calculé pour un index (str)
def _compute_pr_str(index: str) -> str:
    try:
        res = calculer_prix_sql(index)
        # structure attendue: res["tree"]["PR"]
        pr_val = res.get("tree", {}).get("PR", None)
        if pr_val is None:
            return ""
        # format: nombre simple
        return str(pr_val)
    except Exception:
        return ""

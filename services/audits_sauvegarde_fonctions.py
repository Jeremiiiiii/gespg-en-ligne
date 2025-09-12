import os, json, sqlite3
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict, Tuple

from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st

from SQL.db import get_conn,DB_PATH


DB_TS_ARE_UTC: bool = st.session_state.get("DB_TS_ARE_UTC", True)
LOCAL_TZ_NAME: str = st.session_state.get("TZ_NAME", "Europe/Paris")
LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)


ACTION_TITLES_FR = {
    "CREATE_ROW": "CRÉÉ",
    "UPDATE_FIELD": "MODIFIÉ",
    "UPDATE_ROW": "MODIFIÉ",
    "UPSERT_PRODUCT": "MODIFIÉ",
    "DELETE_ROW": "SUPPRIMÉ",
    "DELETE_PRODUCT": "SUPPRIMÉ",
}

# ====== Réglages horaire ======
# Si vos timestamps DB sont déjà en heure locale, mettez ceci à False via session_state.
DB_TS_ARE_UTC: bool = st.session_state.get("DB_TS_ARE_UTC", True)
LOCAL_TZ_NAME: str = st.session_state.get("TZ_NAME", "Europe/Paris")
LOCAL_TZ = ZoneInfo(LOCAL_TZ_NAME)


# ==== Imports utilitaires SQL (colonnes autorisées + upsert standard) ====

from SQL.sql_bom import ALL_PRODUCT_FIELDS, upsert_product_row
_ALLOWED_FIELDS = sorted(set(ALL_PRODUCT_FIELDS) - {"product_index","kind"})

ACTION_TITLES_FR = {
    "CREATE_ROW": "CRÉÉ",
    "UPDATE_FIELD": "MODIFIÉ",
    "UPDATE_ROW": "MODIFIÉ",
    "UPSERT_PRODUCT": "MODIFIÉ",
    "DELETE_ROW": "SUPPRIMÉ",
    "DELETE_PRODUCT": "SUPPRIMÉ",
}

# =================== Helpers Horaire ===================

def _pill_for_action(action: str, count: Optional[int] = None) -> Tuple[str, str]:

    if action == "CREATE_BULK":
        return "create", f"CRÉATIONS MULTIPLES ({count or 0})"
    if action == "DELETE_BULK":
        return "delete", f"SUPPRESSIONS MULTIPLES ({count or 0})"
    if action == "CREATE_ROW":
        return "create", "CRÉÉ"
    if action in ("DELETE_PRODUCT", "DELETE_ROW"):
        return "delete", "SUPPRIMÉ"
    return "rowupd", "MODIFIÉ"



@st.cache_data(ttl=30)
def list_actions() -> List[str]:
    # On lit les actions réellement présentes et on garde un fallback cohérent
    allowed = {
        "UPDATE_FIELD", "UPSERT_PRODUCT", "UPDATE_ROW", "CREATE_ROW",
        "DELETE_PRODUCT", "ADD_EDGE", "REMOVE_EDGE", "ELIMINATION_DOUBLONS",
            "DELETE_ROW"
    }
    try:
        with get_conn() as conn:
            rows = conn.execute("SELECT DISTINCT action FROM audit_log ORDER BY 1;").fetchall()
            vals = [ (r[0] if not isinstance(r, dict) else r["action"]) for r in rows ]
            # Filtrer sur les actions pertinentes si besoin
            vals = [a for a in vals if a in allowed] or list(allowed - {"ELIMINATION_DOUBLONS"})
            return sorted(vals)
    except Exception:
        # Fallback si la lecture DB échoue
        return sorted(list(allowed - {"ELIMINATION_DOUBLONS"}))


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.strptime(ts[:19], "%Y-%m-%d %H:%M:%S")
        if DB_TS_ARE_UTC:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.replace(tzinfo=LOCAL_TZ)  # si la DB stocke déjà en local
        return dt
    except Exception:
        return None

def _ts_local_str(ts: str) -> str:
    """Convertit ts DB -> heure locale Europe/Paris, format yyyy-mm-dd HH:MM:SS"""
    dt = _parse_ts(ts)
    if not dt:
        return (ts or "")[:19]
    return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")

def _ts_minute_local(ts: str) -> str:
    """Retourne la clé minute locale: yyyy-mm-dd HH:MM"""
    dt = _parse_ts(ts)
    if not dt:
        return (ts or "")[:16]
    return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")

# =================== Chargement & actions DB ===================


@st.cache_data(ttl=30)
def load_cell_audit(user: Optional[str], dfrom: Optional[str], dto: Optional[str], limit: int = 1000) -> pd.DataFrame:
    q = """
    SELECT id, ts, user, action, table_name, rec_key,
           field, old_value, new_value, note,
           before_json, after_json
    FROM audit_log
    WHERE action IN (
        'UPDATE_FIELD','UPSERT_PRODUCT','UPDATE_ROW','CREATE_ROW','DELETE_PRODUCT',
        'ADD_EDGE','REMOVE_EDGE','DELETE_ROW'
    )
    AND table_name IN ('products','bom_edges','fournisseurs')
    """
    params: List[Any] = []
    if user and user.strip() and user != "__ALL__":
        q += " AND user = ?"; params.append(user.strip())
    if dfrom:
        q += " AND ts >= ?"; params.append(dfrom + " 00:00:00")
    if dto:
        q += " AND ts <= ?"; params.append(dto + " 23:59:59")
    q += " ORDER BY ts DESC, id DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        return pd.read_sql_query(q, conn, params=params)

def _edge_from_json(js: Optional[str]) -> Dict[str, Any]:
    if not js:
        return {}
    try:
        d = json.loads(js)
        return {
            "parent_index": d.get("parent_index"),
            "child_index": d.get("child_index"),
            "quantity": d.get("quantity"),
        }
    except Exception:
        return {}

def _split_edge_rec_key(rec_key: str) -> Tuple[str, str]:
    if "->" in (rec_key or ""):
        p, c = rec_key.split("->", 1)
        return p.strip(), c.strip()
    return rec_key, ""

def _get_label(ix: str) -> str:
    try:
        with get_conn() as conn:
            r = conn.execute("SELECT libelle_produit FROM products WHERE product_index=?;", (ix,)).fetchone()
            if r and r["libelle_produit"]:
                return str(r["libelle_produit"])
    except Exception:
        pass
    return "—"

def _edge_labels(parent_ix: str, child_ix: str) -> Tuple[str, str]:
    return _get_label(parent_ix), _get_label(child_ix)

@st.cache_data(ttl=30)
def list_known_users() -> List[str]:
    users = set()
    with get_conn() as conn:
        for r in conn.execute("SELECT DISTINCT user FROM audit_log WHERE user IS NOT NULL AND user <> '' ORDER BY 1;"):
            users.add(str(r["user"]))
    try:
        if os.path.exists("data/users.json"):
            with open("data/users.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                for u in data.keys():
                    users.add(str(u))
    except Exception:
        pass
    return sorted(users, key=lambda s: s.lower())

def delete_audit_ids(ids: List[int]) -> None:
    if not ids: return
    with get_conn() as conn:
        sql = f"DELETE FROM audit_log WHERE id IN ({','.join(['?']*len(ids))});"
        conn.execute(sql, ids)
        conn.commit()

def reset_audit_log() -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM audit_log;")
        conn.commit()
    with sqlite3.connect(DB_PATH, isolation_level=None) as vconn:
        vconn.execute("VACUUM;")

# =================== Helpers Diff ===================

def _diff_from_json(before_json: Optional[str], after_json: Optional[str]) -> List[Dict[str, str]]:
    try:
        before = json.loads(before_json) if before_json else {}
    except Exception:
        before = {}
    try:
        after = json.loads(after_json) if after_json else {}
    except Exception:
        after = {}
    ignore = {"product_index", "kind"}
    keys = sorted(set(before.keys()) | set(after.keys()))
    out = []
    for k in keys:
        if k in ignore: 
            continue
        old = "" if before.get(k) is None else str(before.get(k))
        new = "" if after.get(k) is None else str(after.get(k))
        if old != new:
            out.append({"field": k, "old": old, "new": new})
    return out

def _get_libelle(rec_key: str, before_json: Optional[str] = None, after_json: Optional[str] = None) -> str:
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT libelle_produit FROM products WHERE product_index=?;",
                (rec_key,)
            ).fetchone()
            if row and (row["libelle_produit"] or "").strip():
                return str(row["libelle_produit"])
    except Exception:
        pass
    # fallback JSON : libellé produit OU nom fournisseur
    for js in (after_json, before_json):
        if js:
            try:
                d = json.loads(js)
                v = d.get("libelle_produit") or d.get("nom")
                if v:
                    return str(v)
            except Exception:
                continue
    # dernier repli : la clé elle-même
    return rec_key or "—"

def _supplier_name(rec_key: str, before_json: Optional[str] = None, after_json: Optional[str] = None) -> str:
    # essaie de lire 'nom' dans les snapshots; sinon rec_key
    for js in (after_json, before_json):
        if js:
            try:
                d = json.loads(js) or {}
                if d.get("nom"):
                    return str(d["nom"])
            except Exception:
                pass
    return rec_key or "—"


def _apply_update_fields_fournisseur(nom: str, changes: list[dict]) -> None:
    if not changes: return
    sets = []
    params = []
    for ch in changes:
        field = ch.get("field")
        new   = ch.get("new")
        if field in ("categorie","adresse","code_postal","ville","pays"):
            sets.append(f"{field}=?")
            params.append(new)
    if sets:
        params.append(nom)
        with get_conn() as conn:
            conn.execute(f"UPDATE fournisseurs SET {', '.join(sets)} WHERE nom=?;", params)
            conn.commit()

def _apply_upsert_fournisseur_before(before_json: Optional[str], nom: str) -> None:
    before = json.loads(before_json) if before_json else None
    with get_conn() as conn:
        if before is None:
            conn.execute("DELETE FROM fournisseurs WHERE nom=?;", (nom,))
        else:
            conn.execute("""
                INSERT INTO fournisseurs (nom, categorie, adresse, code_postal, ville, pays)
                VALUES (:nom, :categorie, :adresse, :code_postal, :ville, :pays)
                ON CONFLICT(nom) DO UPDATE SET
                    categorie=excluded.categorie, adresse=excluded.adresse,
                    code_postal=excluded.code_postal, ville=excluded.ville, pays=excluded.pays;
            """, before)
        conn.commit()

def _apply_remove_created_fournisseur(nom: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM fournisseurs WHERE nom=?;", (nom,))
        conn.commit()

def _apply_restore_deleted_fournisseur(before_json: Optional[str]) -> None:
    before = json.loads(before_json) if before_json else None
    if not before: return
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO fournisseurs (nom, categorie, adresse, code_postal, ville, pays)
            VALUES (:nom, :categorie, :adresse, :code_postal, :ville, :pays)
            ON CONFLICT(nom) DO UPDATE SET
                categorie=excluded.categorie, adresse=excluded.adresse,
                code_postal=excluded.code_postal, ville=excluded.ville, pays=excluded.pays;
        """, before)
        conn.commit()

# =================== Grouping (user/rec_key/seconde) ===================

def _build_groups(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.to_dict(orient="records")
    groups: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    unified: List[Dict[str, Any]] = []

    for r in records:
        action = r.get("action")
        table = r.get("table_name")
        ts_local = _ts_local_str(r.get("ts", ""))
        ts_minute = _ts_minute_local(r.get("ts", ""))
        user = r.get("user") or "—"
        rec_key = r.get("rec_key") or "—"

        # === PRODUCTS ===
        if table == "products":
            if action == "UPDATE_FIELD":
                key = (table, ts_local, user, rec_key)  # par seconde + record
                g = groups.get(key)
                if g is None:
                    g = {
                        "table": table,
                        "ts_sec": ts_local, "ts_min": ts_minute, "user": user, "rec_key": rec_key,
                        "action": "UPDATE_FIELD",
                        "changes": [], "ids": [],
                        "label": _get_libelle(rec_key),
                        "before_json": None, "after_json": None,
                    }
                    groups[key] = g
                g["changes"].append({
                    "field": r.get("field") or "",
                    "old": r.get("old_value") or "",
                    "new": r.get("new_value") or "",
                })
                g["ids"].append(int(r["id"]))

            elif action in ("UPSERT_PRODUCT","UPDATE_ROW"):
                changes = _diff_from_json(r.get("before_json"), r.get("after_json"))
                unified.append({
                    "table": table,
                    "ts_sec": ts_local, "ts_min": ts_minute, "user": user, "rec_key": rec_key,
                    "action": "UPDATE_ROW",
                    "changes": changes,
                    "ids": [int(r["id"])],
                    "label": _get_libelle(rec_key, r.get("before_json"), r.get("after_json")),
                    "before_json": r.get("before_json"),
                    "after_json": r.get("after_json"),
                })

            elif action == "CREATE_ROW":
                changes = _diff_from_json(r.get("before_json"), r.get("after_json"))
                unified.append({
                    "table": table,
                    "ts_sec": ts_local, "ts_min": ts_minute, "user": user, "rec_key": rec_key,
                    "action": "CREATE_ROW",
                    "changes": changes,
                    "ids": [int(r["id"])],
                    "label": _get_libelle(rec_key, r.get("before_json"), r.get("after_json")),
                    "before_json": r.get("before_json"),
                    "after_json": r.get("after_json"),
                })

            elif action == "DELETE_PRODUCT":
                unified.append({
                    "table": table,
                    "ts_sec": ts_local, "ts_min": ts_minute, "user": user, "rec_key": rec_key,
                    "action": "DELETE_PRODUCT",
                    "changes": [],
                    "ids": [int(r["id"])],
                    "label": _get_libelle(rec_key, r.get("before_json"), None),
                    "before_json": r.get("before_json"),
                    "after_json": r.get("after_json"),
                })

        # === BOM_EDGES ===
        elif table == "bom_edges":
            parent_ix, child_ix = _split_edge_rec_key(rec_key)
            parent_label, child_label = _edge_labels(parent_ix, child_ix)

            if action == "ADD_EDGE":
                bef = _edge_from_json(r.get("before_json"))
                aft = _edge_from_json(r.get("after_json"))
                old_q = "" if not bef else ("" if bef.get("quantity") is None else str(bef.get("quantity")))
                new_q = "" if not aft else ("" if aft.get("quantity") is None else str(aft.get("quantity")))
                sem_action = "CREATE_ROW" if not bef else "UPDATE_ROW"

                unified.append({
                    "table": table,
                    "ts_sec": ts_local, "ts_min": ts_minute, "user": user, "rec_key": rec_key,
                    "action": sem_action,
                    "changes": [{"field": "quantity", "old": old_q, "new": new_q}],
                    "ids": [int(r["id"])],
                    "label": f"{parent_label} → {child_label}",
                    "before_json": r.get("before_json"),
                    "after_json": r.get("after_json"),
                    "parent_index": parent_ix,
                    "child_index": child_ix,
                })

            elif action == "REMOVE_EDGE":
                bef = _edge_from_json(r.get("before_json"))
                old_q = "" if not bef else ("" if bef.get("quantity") is None else str(bef.get("quantity")))
                unified.append({
                    "table": table,
                    "ts_sec": ts_local, "ts_min": ts_minute, "user": user, "rec_key": rec_key,
                    "action": "DELETE_PRODUCT",
                    "changes": [{"field": "quantity", "old": old_q, "new": ""}],
                    "ids": [int(r["id"])],
                    "label": f"{parent_label} → {child_label}",
                    "before_json": r.get("before_json"),
                    "after_json": r.get("after_json"),
                    "parent_index": parent_ix,
                    "child_index": child_ix,
                })
        elif table == "fournisseurs":
            if action == "UPDATE_FIELD":
                key = (table, ts_local, user, rec_key)
                g = groups.get(key)
                if g is None:
                    g = {
                        "table": table,
                        "ts_sec": ts_local, "ts_min": ts_minute, "user": user, "rec_key": rec_key,
                        "action": "UPDATE_FIELD",
                        "changes": [], "ids": [],
                        "label": rec_key,  # libellé = nom fournisseur
                        "before_json": None, "after_json": None,
                    }
                    groups[key] = g
                g["changes"].append({
                    "field": r.get("field"), "old": r.get("old_value"), "new": r.get("new_value")
                })
                g["ids"].append(int(r["id"]))

            elif action in ("UPDATE_ROW", "CREATE_ROW", "DELETE_ROW"):
                changes = _diff_from_json(r.get("before_json"), r.get("after_json"))
                unified.append({
                    "table": table,
                    "ts_sec": ts_local, "ts_min": ts_minute, "user": user, "rec_key": rec_key,
                    "action": action,
                    "changes": changes,
                    "ids": [int(r["id"])],
                    "label": rec_key,
                    "before_json": r.get("before_json"),
                    "after_json": r.get("after_json"),
                })

    unified.extend(groups.values())
    unified.sort(key=lambda x: (x["ts_sec"], x.get("id", 0)), reverse=True)
    # coalescer multi-créations/suppressions (>=4) par minute et utilisateur
    return _coalesce_bulk(unified, threshold=3)

def _coalesce_bulk(items: List[Dict[str, Any]], threshold: int = 3) -> List[Dict[str, Any]]:
    """
    Regroupe CREATE_ROW et DELETE_PRODUCT par (user, ts_min) si count >= threshold.
    Concerne toutes les tables (products + bom_edges).
    """
    # Index par (action, user, minute)
    buckets: Dict[Tuple[str, str, str], List[int]] = {}
    for idx, g in enumerate(items):
        act = g.get("action")
        if act not in ("CREATE_ROW", "DELETE_PRODUCT","DELETE_ROW"):
            continue
        key = (act, g.get("user"), g.get("ts_min"))
        buckets.setdefault(key, []).append(idx)

    to_hide: set[int] = set()
    bulk_items: List[Dict[str, Any]] = []

    for (act, user, minute), idxs in buckets.items():
        if len(idxs) >= threshold:
            # Construire un bloc "bulk"
            children = [items[i] for i in idxs]
            ids: List[int] = []
            for c in children:
                ids.extend(c.get("ids", []))
            bulk_items.append({
                "table": "mixed",
                "ts_sec": f"{minute}:00",   # affichage minute
                "ts_min": minute,
                "user": user,
                "action": "CREATE_BULK" if act == "CREATE_ROW" else "DELETE_BULK",
                "count": len(children),
                "children": children,
                "ids": ids,
                "label": "",  # non utilisé
            })
            to_hide.update(idxs)

    # Conserver les items non engloutis + ajouter les bulk
    out = [g for i, g in enumerate(items) if i not in to_hide]
    out.extend(bulk_items)
    out.sort(key=lambda x: x["ts_sec"], reverse=True)
    return out

# =================== UNDO (revenir en arrière) ===================

def _apply_upsert_before(before_json: Optional[str], rec_key: str) -> None:
    before = json.loads(before_json) if before_json else None
    with get_conn() as conn:
        if before is None:
            conn.execute("DELETE FROM products WHERE product_index=?;", (rec_key,))
        else:
            upsert_product_row(conn, before, user=None, audit=False)
        conn.commit()

def _apply_restore_deleted(before_json: Optional[str], rec_key: str) -> None:
    before = json.loads(before_json) if before_json else None
    if not before:
        return
    with get_conn() as conn:
        upsert_product_row(conn, before, user=None, audit=False)
        conn.commit()

def _apply_update_fields(rec_key: str, changes: List[Dict[str, str]]) -> None:
    up: Dict[str, Any] = {}
    for ch in changes:
        f = ch.get("field")
        if f not in _ALLOWED_FIELDS:
            continue
        old = ch.get("old", "")
        up[f] = None if (old is None or str(old) == "") else str(old)
    if not up:
        return
    sets = ", ".join([f"{k} = :{k}" for k in up.keys()])
    up["ix"] = rec_key
    with get_conn() as conn:
        conn.execute(f"UPDATE products SET {sets} WHERE product_index=:ix;", up)
        conn.commit()

def _apply_remove_created(rec_key: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM products WHERE product_index=?;", (rec_key,))
        conn.commit()

def _undo_edge_add_or_update(before_json: Optional[str], after_json: Optional[str], rec_key: str) -> None:
    before = _edge_from_json(before_json)
    _ = _edge_from_json(after_json)
    parent_ix, child_ix = _split_edge_rec_key(rec_key)
    with get_conn() as conn:
        if not before or not before.get("parent_index"):
            conn.execute("DELETE FROM bom_edges WHERE parent_index=? AND child_index=?;", (parent_ix, child_ix))
        else:
            q = before.get("quantity")
            conn.execute(
                "INSERT INTO bom_edges(parent_index, child_index, quantity) VALUES (?,?,?) "
                "ON CONFLICT(parent_index, child_index) DO UPDATE SET quantity=excluded.quantity;",
                (parent_ix, child_ix, q)
            )
        conn.commit()

def _undo_edge_remove(before_json: Optional[str], rec_key: str) -> None:
    before = _edge_from_json(before_json)
    parent_ix, child_ix = _split_edge_rec_key(rec_key)
    if not before:
        return
    q = before.get("quantity")
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO bom_edges(parent_index, child_index, quantity) VALUES (?,?,?) "
            "ON CONFLICT(parent_index, child_index) DO UPDATE SET quantity=excluded.quantity;",
            (parent_ix, child_ix, q)
        )
        conn.commit()

def revert_group(g: Dict[str, Any]) -> Tuple[bool, str]:
    """Applique le retour en arrière pour un bloc (produits ET liaisons). Gère aussi les blocs multiples."""
    # Bloc multiple
    if g.get("action") in ("CREATE_BULK", "DELETE_BULK"):
        oks, msgs = [], []
        for child in g.get("children", []):
            ok, msg = revert_group(child)
            oks.append(ok); msgs.append(msg)
        all_ok = all(oks) if oks else True
        if all_ok:
            return True, f"Retour en arrière appliqué pour {len(g.get('children', []))} éléments ✅"
        return False, "Certaines annulations ont échoué : " + " ; ".join([m for m in msgs if m])

    try:
        table = g.get("table", "products")
        act = g.get("action")

        if table == "products":
            if act == "UPDATE_FIELD":
                _apply_update_fields(g["rec_key"], g.get("changes", []) or [])
            elif act == "UPDATE_ROW":
                _apply_upsert_before(g.get("before_json"), g["rec_key"])
            elif act == "CREATE_ROW":
                _apply_remove_created(g["rec_key"])
            elif act == "DELETE_PRODUCT":
                _apply_restore_deleted(g.get("before_json"), g["rec_key"])
            else:
                return False, "Action non supportée (produits)."

        elif table == "bom_edges":
            if act in ("UPDATE_ROW", "CREATE_ROW"):
                _undo_edge_add_or_update(g.get("before_json"), g.get("after_json"), g["rec_key"])
            elif act == "DELETE_PRODUCT":
                _undo_edge_remove(g.get("before_json"), g["rec_key"])
            else:
                return False, "Action non supportée (liaisons)."
            
        elif table == "fournisseurs":
            if act == "UPDATE_FIELD":
                _apply_update_fields_fournisseur(g["rec_key"], g.get("changes", []) or [])
            elif act == "UPDATE_ROW":
                _apply_upsert_fournisseur_before(g.get("before_json"), g["rec_key"])
            elif act == "CREATE_ROW":
                _apply_remove_created_fournisseur(g["rec_key"])
            elif act == "DELETE_ROW":
                _apply_restore_deleted_fournisseur(g.get("before_json"))
            return True, "Retour en arrière appliqué (fournisseur) ✅"

        else:
            return False, "Table inconnue."
        


        return True, "Retour en arrière appliqué ✅"
    except sqlite3.IntegrityError as e:
        return False, f"Impossible d'annuler (contrainte d'intégrité) : {e}"
    except Exception as e:
        return False, f"Échec de l'annulation : {e}"

def _ncols_for_changes(n: int) -> int:
    """
    Nombre de colonnes pour afficher les changements.
    1–4  -> 2 colonnes
    5–11 -> 3 colonnes
    12+  -> 4 colonnes
    """
    if n <= 1:
        return 1
    if n <= 4:
        return 2
    if n >= 12:
        return 4
    return 3
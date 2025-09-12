#sql_bom

from __future__ import annotations
import sqlite3
from typing import List, Dict, Tuple, Optional, Union,Any
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import re
import json
import pandas as pd  
import os


from SQL.db import get_conn 

# ================ Normalisation & Utils ===============

def _normalize_index_value(x: str) -> str:
    """Trim + uppercase pour les index produits."""
    return (str(x or "")).strip().upper()

def _clean_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s != "" else None

_EURO_RE = re.compile(r"[ €\u202f]")  # supprime symbole euro et espaces fines
def _to_decimal_str(val: Union[str, float, int, None], scale: int) -> Optional[str]:
    """
    Parse des chaînes euro/fr vers Decimal avec 'scale' décimales.
    Retourne une chaîne normalisée ('15.48') ou None si vide.
    """
    if val is None:
        return None
    s = str(val).strip()
    if s == "":
        return None
    # Nettoyage basique
    s = _EURO_RE.sub("", s)        # retire '€' et espaces
    s = s.replace(",", ".")        # virgule -> point
    s = s.replace('"', "").replace("'", "")
    if s in ("-", "--"):
        return None
    try:
        q = Decimal(s)
    except Exception:
        s2 = re.sub(r"[^0-9\.\-]", "", s)
        if s2 == "" or s2 in ("-", "--"):
            return None
        try:
            q = Decimal(s2)
        except InvalidOperation:
            return None
    quant = Decimal("1").scaleb(-scale)  # 10^-scale
    q = q.quantize(quant, rounding=ROUND_HALF_UP)
    return format(q, "f")

def _to_int_bool(val: Union[str, int, float, None]) -> Optional[int]:
    """Retourne 0/1 pour diverses entrées (Oui/Non, True/False, 0/1). None si vide."""
    if val is None:
        return None
    s = str(val).strip().lower()
    if s == "":
        return None
    if s in ("1", "true", "vrai", "yes", "oui", "y", "t"):
        return 1
    if s in ("0", "false", "faux", "no", "non", "n", "f"):
        return 0
    try:
        return 1 if int(float(s)) != 0 else 0
    except Exception:
        return None



#=====================Tracabillité===================

def log_audit(conn: sqlite3.Connection,
              user: Optional[str],
              action: str,
              table: str,
              rec_key: str,
              field: Optional[str] = None,
              old_value: Optional[Any] = None,
              new_value: Optional[Any] = None,
              note: Optional[str] = None,
              before_obj: Optional[Dict[str, Any]] = None,
              after_obj: Optional[Dict[str, Any]] = None) -> None:
    """
    Journal polyvalent :
    - niveau cellule : utiliser field / old_value / new_value
    - niveau ligne/edge : utiliser before_obj / after_obj (sérialisés en JSON)
    Les deux styles peuvent coexister ; les champs non utilisés restent NULL.
    """
    init_audit_schema(conn)

    def _to_str(x: Any) -> Optional[str]:
        if x is None:
            return None
        return str(x)

    before_json = json.dumps(before_obj, ensure_ascii=False) if before_obj is not None else None
    after_json  = json.dumps(after_obj,  ensure_ascii=False) if after_obj  is not None else None

    conn.execute(
        """
        INSERT INTO audit_log(
            user, action, table_name, rec_key,
            field, old_value, new_value, note,
            before_json, after_json
        )
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (user, action, table, rec_key,
         field, _to_str(old_value), _to_str(new_value), note,
         before_json, after_json)
    )
    
def _table_cols(conn: sqlite3.Connection, table: str) -> set:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return {r["name"] for r in rows}

def init_audit_schema(conn: sqlite3.Connection) -> None:
    # table de base (si première fois)
    
    conn.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT NOT NULL DEFAULT (datetime('now')),
      user TEXT,
      action TEXT NOT NULL,
      table_name TEXT NOT NULL,
      rec_key TEXT NOT NULL,
      before_json TEXT,
      after_json TEXT,
      note TEXT
    );
    """)
    # migration -> colonnes cellule
    cols = _table_cols(conn, "audit_log")
    if "field" not in cols:
        conn.execute("ALTER TABLE audit_log ADD COLUMN field TEXT;")
    if "old_value" not in cols:
        conn.execute("ALTER TABLE audit_log ADD COLUMN old_value TEXT;")
    if "new_value" not in cols:
        conn.execute("ALTER TABLE audit_log ADD COLUMN new_value TEXT;")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_table ON audit_log(table_name);")
    conn.commit()
    

# maps d'en-têtes UI -> colonnes DB (tu les as déjà)
# PF_MAP / SF_MAP / _canonicalize_row(...) sont déjà dans ton fichier

_PRODUCT_FIELDS_ORDER = [
    # ordre neutre ; on exclut product_index/kind du diff
    "reference","libelle_produit","composition","couleur","marque","famille","libelle_famille",
    "unite","code_liaison_externe","commentaire","prix_achat","pr","pv_ttc","fournisseur",
    "designation_fournisseur","compose"
]

def _row_to_displayable(field: str, val: Any) -> str:
    # pour l’audit (affichage ancien → nouveau)
    if val is None: return ""
    return str(val)
# --- Normalisation pour comparaison cellule par cellule (diff audit) ---
_NUM_FIELDS = {"prix_achat", "pr", "pv_ttc"}  # champs à 2 décimales

def _norm_for_compare(field: str, val: Any) -> str:
    """
    Retourne une chaîne 'normalisée' pour comparer l'ancien et le nouveau :
    - trim des chaînes
    - pour les prix/PR/PV, arrondi à 2 décimales et format '12.34'
    - None -> ""
    """
    if val is None:
        return ""
    s = str(val).strip()
    if s == "":
        return ""
    if field in _NUM_FIELDS:
        from decimal import Decimal, ROUND_HALF_UP
        s2 = s.replace("€", "").replace("\u202f", "").replace(" ", "").replace(",", ".")
        try:
            q = Decimal(s2).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            return format(q, "f")
        except Exception:
            return ""
    return s

def upsert_products_diff(conn: sqlite3.Connection, df_pf, df_sf, user: Optional[str]=None) -> Dict[str,int]:
    """
    Compare SQL.products avec DF et n'applique/log QUE les cellules modifiées.
    - création ligne -> log géré par upsert_product_row (pas ici)
    - suppression ligne -> à faire ailleurs (delete_product)
    - modification cellule -> 1 log UPDATE_FIELD par cellule
    """
    if pd is None:
        raise RuntimeError("pandas n'est pas disponible.")
    init_schema(conn)

    changed_cells = 0
    created_rows = 0

    def _apply_rows(df: pd.DataFrame, kind: str):
        nonlocal changed_cells, created_rows
        if df is None or "Index" not in df.columns:
            return

        for _, r in df.fillna("").iterrows():
            row_ui = r.to_dict()
            canon = _canonicalize_row(row_ui, PF_MAP if kind == "PF" else SF_MAP, kind)

            ix = canon["product_index"]
            cur = conn.execute("SELECT * FROM products WHERE product_index=?", (ix,)).fetchone()

            if cur is None:
                # Création : on laisse upsert_product_row journaliser CREATE_ROW
                upsert_product_row(conn, canon, user=user)
                created_rows += 1
                continue

            # Modification cellule par cellule
            cur_d = dict(cur)
            updates: Dict[str, Any] = {}
            for f in _PRODUCT_FIELDS_ORDER:
                if f not in canon:
                    continue
                old = _norm_for_compare(f, cur_d.get(f))
                new = _norm_for_compare(f, canon.get(f))
                if old != new:
                    updates[f] = canon.get(f)
                    log_audit(
                        conn, user=user, action="UPDATE_FIELD", table="products",
                        rec_key=ix, field=f,
                        old_value=_row_to_displayable(f, cur_d.get(f)),
                        new_value=_row_to_displayable(f, canon.get(f)),
                        note=f"kind={kind}"
                    )
                    changed_cells += 1

            if updates:
                sets = ", ".join([f"{k} = :{k}" for k in updates.keys()])
                updates["ix"] = ix
                conn.execute(f"UPDATE products SET {sets} WHERE product_index=:ix;", updates)

    with conn:
        if df_pf is not None:
            _apply_rows(df_pf, "PF")
        if df_sf is not None:
            if "PA" in df_sf.columns and "Prix d'achat" not in df_sf.columns:
                df_sf = df_sf.copy()
                df_sf["Prix d'achat"] = df_sf["PA"]
            _apply_rows(df_sf, "SF")

    return {"changed_cells": changed_cells, "created_rows": created_rows}

# =================== Schéma (idempotent) ==============


# Modèle : une table 'products' avec 'kind' ('PF' ou 'SF') + attributs communs,
# et quelques champs spécifiques SF (fournisseur, designation_fournisseur, compose).
# La table bom_edges existante est conservée ; un enfant 'PF' est interdit (CHECK).
CREATE_FOURNISSEUR_TABLE= """
    CREATE TABLE IF NOT EXISTS fournisseurs (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        nom             TEXT NOT NULL UNIQUE,
        categorie       TEXT,
        adresse         TEXT,
        code_postal     TEXT,
        ville           TEXT,
        pays            TEXT
    );
    """
CREATE_PRODUCTS_SQL = """
CREATE TABLE IF NOT EXISTS products (
  product_index TEXT PRIMARY KEY,
  kind TEXT CHECK (kind IN ('PF','SF')),
  reference TEXT,
  libelle_produit TEXT,
  composition TEXT,
  couleur TEXT,
  marque TEXT,
  famille TEXT,
  libelle_famille TEXT,
  unite TEXT,
  code_liaison_externe TEXT,
  commentaire TEXT,
  prix_achat NUMERIC,   -- 2 décimales (arrondi HALF_UP)
  pr NUMERIC,           -- 2 décimales (arrondi HALF_UP)
  pv_ttc NUMERIC,       -- 2 décimales
  fournisseur TEXT,
  designation_fournisseur TEXT,
  compose INTEGER CHECK (compose IN (0,1))
);
"""

CREATE_BOM_EDGES_SQL = """
CREATE TABLE IF NOT EXISTS bom_edges (
  parent_index TEXT NOT NULL,
  child_index  TEXT NOT NULL,
  quantity     NUMERIC NOT NULL CHECK (quantity >= 0),
  PRIMARY KEY (parent_index, child_index),
  FOREIGN KEY (parent_index) REFERENCES products(product_index) ON DELETE RESTRICT ON UPDATE CASCADE,
  FOREIGN KEY (child_index)  REFERENCES products(product_index) ON DELETE RESTRICT ON UPDATE CASCADE,
  CHECK (substr(child_index, 1, 2) <> 'PF')
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_bom_edges_child ON bom_edges(child_index);
CREATE INDEX IF NOT EXISTS idx_products_kind ON products(kind);
CREATE INDEX IF NOT EXISTS idx_products_famille ON products(famille);
"""

CREATE_PREFS_SQL = """
CREATE TABLE IF NOT EXISTS user_prefs (
  pref_key   TEXT PRIMARY KEY,
  pref_value TEXT,
  updated_at TEXT DEFAULT (datetime('now'))
);
"""
# --- Compatibilité ancienne API ---
def sync_products_from_csv(conn, df_pf=None, df_sf=None):
    # ancien nom -> nouveau comportement
    from .sql_bom import import_products_from_dataframes as _impl  # si module relatif
    return _impl(conn, df_pf, df_sf)

def compute_bom_tree(conn, root_index, *_, **__):
    # ignorer df_fini/df_semi ; lire depuis SQL désormais
    from .sql_bom import compute_bom_tree_sql as _impl
    return _impl(conn, root_index)

def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [r["name"] for r in rows]

def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")

    # 1) Créer les tables si elles n'existent pas (ne casse rien d'existant)
    conn.executescript(CREATE_FOURNISSEUR_TABLE)
    conn.executescript(CREATE_PRODUCTS_SQL)
    conn.executescript(CREATE_BOM_EDGES_SQL)
    conn.executescript(CREATE_PREFS_SQL)  
    conn.commit()
    # 2) Ajouter les colonnes manquantes sur 'products' si base ancienne
    wanted_cols = {
        "kind": "TEXT",
        "reference": "TEXT",
        "libelle_produit": "TEXT",
        "composition": "TEXT",
        "couleur": "TEXT",
        "marque": "TEXT",
        "famille": "TEXT",
        "libelle_famille": "TEXT",
        "unite": "TEXT",
        "code_liaison_externe": "TEXT",
        "commentaire": "TEXT",
        "prix_achat": "NUMERIC",
        "pr": "NUMERIC",
        "pv_ttc": "NUMERIC",
        "fournisseur": "TEXT",
        "designation_fournisseur": "TEXT",
        "compose": "INTEGER"
    }
    existing = set(_table_columns(conn, "products"))
    for col, typ in wanted_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE products ADD COLUMN {col} {typ};")

    # 3) Créer les index **après** l'ajout des colonnes (et seulement s'ils sont pertinents)
    #    (évite: OperationalError: no such column: kind)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bom_edges_child ON bom_edges(child_index);")
    existing = set(_table_columns(conn, "products"))  # re-scan après ALTER
    if "kind" in existing:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_kind ON products(kind);")
    if "famille" in existing:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_products_famille ON products(famille);")
    init_audit_schema(conn)
    conn.commit()

def get_pref(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT pref_value FROM user_prefs WHERE pref_key=?;", (key,)).fetchone()
    return None if row is None else row[0]

def set_pref(conn: sqlite3.Connection, key: str, value: str, user: Optional[str] = None) -> None:
    with conn:
        conn.execute("""
        INSERT INTO user_prefs(pref_key, pref_value)
        VALUES(?, ?)
        ON CONFLICT(pref_key) DO UPDATE SET
          pref_value = excluded.pref_value,
          updated_at = datetime('now');
        """, (key, value))
        # (optionnel) journaliser dans audit_log si tu veux tracer
        try:
            log_audit(conn, user=user, action="set_pref", table="user_prefs",
                      rec_key=key, field="pref_value", new_value=value)
        except Exception:
            pass

# ===================== Upserts ========================


UPSERT_PRODUCT_SQL = """
INSERT INTO products (
    product_index, kind, reference, libelle_produit, composition, couleur,
    marque, famille, libelle_famille, unite, code_liaison_externe, commentaire,
    prix_achat, pr, pv_ttc, fournisseur, designation_fournisseur, compose
) VALUES (
    :product_index, :kind, :reference, :libelle_produit, :composition, :couleur,
    :marque, :famille, :libelle_famille, :unite, :code_liaison_externe, :commentaire,
    :prix_achat, :pr, :pv_ttc, :fournisseur, :designation_fournisseur, :compose
)
ON CONFLICT(product_index) DO UPDATE SET
    kind = COALESCE(excluded.kind, products.kind),
    reference = COALESCE(excluded.reference, products.reference),
    libelle_produit = COALESCE(excluded.libelle_produit, products.libelle_produit),
    composition = COALESCE(excluded.composition, products.composition),
    couleur = COALESCE(excluded.couleur, products.couleur),
    marque = COALESCE(excluded.marque, products.marque),
    famille = COALESCE(excluded.famille, products.famille),
    libelle_famille = COALESCE(excluded.libelle_famille, products.libelle_famille),
    unite = COALESCE(excluded.unite, products.unite),
    code_liaison_externe = COALESCE(excluded.code_liaison_externe, products.code_liaison_externe),
    commentaire = COALESCE(excluded.commentaire, products.commentaire),
    prix_achat = COALESCE(excluded.prix_achat, products.prix_achat),
    pr = COALESCE(excluded.pr, products.pr),
    pv_ttc = COALESCE(excluded.pv_ttc, products.pv_ttc),
    fournisseur = COALESCE(excluded.fournisseur, products.fournisseur),
    designation_fournisseur = COALESCE(excluded.designation_fournisseur, products.designation_fournisseur),
    compose = COALESCE(excluded.compose, products.compose);
"""
# Toutes les colonnes attendues par l'UPSERT products, dans l'ordre
ALL_PRODUCT_FIELDS = [
    "product_index", "kind",
    "reference", "libelle_produit", "composition", "couleur",
    "marque", "famille", "libelle_famille", "unite",
    "code_liaison_externe", "commentaire",
    "prix_achat", "pr", "pv_ttc",
    "fournisseur", "designation_fournisseur", "compose"
]
def upsert_product_row(conn: sqlite3.Connection, row: Dict[str, object], user: Optional[str]=None, *, audit: bool = True) -> None:
    payload = {k: None for k in ALL_PRODUCT_FIELDS}
    payload.update({k: row.get(k) for k in row.keys() if k in payload})

    if not payload.get("product_index"):
        raise ValueError("product_index manquant pour l'UPSERT.")
    if payload.get("kind") not in ("PF", "SF", None):
        raise ValueError(f"kind invalide: {payload.get('kind')} (attendu: 'PF' ou 'SF').")

    # état avant
    before_row = conn.execute(
        "SELECT * FROM products WHERE product_index=?;",
        (payload["product_index"],)
    ).fetchone()
    before_obj = dict(before_row) if before_row else None

    # upsert
    conn.execute(UPSERT_PRODUCT_SQL, payload)

    # état après
    after_row = conn.execute(
        "SELECT * FROM products WHERE product_index=?;",
        (payload["product_index"],)
    ).fetchone()
    after_obj = dict(after_row) if after_row else None

    if not audit:
        return  # ← Pas de log en mode UNDO

    # Journalisation normale
    if before_obj is None:
        log_audit(
            conn,
            user=user,
            action="CREATE_ROW",
            table="products",
            rec_key=str(payload["product_index"]),
            before_obj=None,
            after_obj=after_obj,
            note=None
        )
    else:
        if before_obj != after_obj:
            log_audit(
                conn,
                user=user,
                action="UPDATE_ROW",
                table="products",
                rec_key=str(payload["product_index"]),
                before_obj=before_obj,
                after_obj=after_obj,
                note=None
            )



# ======================================================
# ============== CSV/DataFrame Migration ===============
# ======================================================

# Mapping des en-têtes 
PF_MAP = {
    "Index":"product_index",
    "Référence":"reference",
    "Libellé produit":"libelle_produit",
    "Composition":"composition",
    "Couleur":"couleur",
    "Marque":"marque",
    "Famille":"famille",
    "Libellé famille":"libelle_famille",
    "Unité":"unite",
    "Code liaison externe":"code_liaison_externe",
    "Commentaire":"commentaire",
    "Prix d'achat":"prix_achat",
    "PR":"pr",
    "PV TTC":"pv_ttc",
}
SF_MAP = {
    "Index":"product_index",
    "Référence":"reference",
    "Libellé produit":"libelle_produit",
    "Composition":"composition",
    "Couleur":"couleur",
    "Marque":"marque",
    "Famille":"famille",
    "Libellé famille":"libelle_famille",
    "Unité":"unite",
    "Code liaison externe":"code_liaison_externe",
    "Commentaire":"commentaire",
    "Prix d'achat":"prix_achat",  # parfois présent
    "PA":"prix_achat",            # variante d'entête
    "PR":"pr",
    "PV TTC":"pv_ttc",
    "Fournisseur":"fournisseur",
    "Désignation fournisseur":"designation_fournisseur",
    "Composé":"compose",
    # ignorés : "ArbreParenté", "Visible", "Quantité:pour"
}

def _canonicalize_row(csv_row: Dict[str, object], colmap: Dict[str,str], kind: str) -> Dict[str, object]:
    """Transforme une ligne CSV en dict canonique pour 'products' (parse nombres, nettoie textes)."""
    out: Dict[str, object] = {
        "product_index": _normalize_index_value(csv_row.get("Index","")) if "Index" in csv_row else None,
        "kind": kind
    }
    # champs texte
    for src, dst in colmap.items():
        if dst in ("prix_achat","pr","pv_ttc","compose","product_index"):
            continue  # gérés ci-dessous / déjà positionnés
        v = _clean_text(csv_row.get(src))
        if v is not None:
            out[dst] = v

    # champs numériques (2 décimales)
    pa_src = "Prix d'achat" if "Prix d'achat" in csv_row else ("PA" if "PA" in csv_row else None)
    if pa_src:
        out["prix_achat"] = _to_decimal_str(csv_row.get(pa_src), scale=2)
    if "PR" in csv_row:
        out["pr"] = _to_decimal_str(csv_row.get("PR"), scale=2)
    if "PV TTC" in csv_row:
        out["pv_ttc"] = _to_decimal_str(csv_row.get("PV TTC"), scale=2)

    # compose (0/1) seulement pour SF
    if kind == "SF":
        comp = None
        if "Composé" in csv_row:
            comp = _to_int_bool(csv_row.get("Composé"))
        out["compose"] = comp

    if not out.get("product_index"):
        raise ValueError("Ligne CSV sans 'Index' valide.")
    return out

def import_products_from_dataframes(conn: sqlite3.Connection, df_pf, df_sf, user: Optional[str]=None) -> Dict[str,int]:
    init_schema(conn)
    total_pf = total_sf = 0
    with conn:
        if df_pf is not None and "Index" in df_pf.columns:
            for _, r in df_pf.fillna("").iterrows():
                row = _canonicalize_row(r.to_dict(), PF_MAP, "PF")
                upsert_product_row(conn, row, user=user)
                total_pf += 1
        if df_sf is not None and "Index" in df_sf.columns:
            for _, r in df_sf.fillna("").iterrows():
                row = _canonicalize_row(r.to_dict(), SF_MAP, "SF")
                upsert_product_row(conn, row, user=user)
                total_sf += 1
    if (total_pf + total_sf) > 0:
        # entrée agrégée optionnelle
        log_audit(
                    conn,
                    user=user,
                    action="IMPORT_PRODUCTS",
                    table="products",
                    rec_key=f"batch:{total_pf}+{total_sf}",
                    note="batch import"
                )

    return {"pf_rows": total_pf, "sf_rows": total_sf}


def import_products_from_csv_paths(conn: sqlite3.Connection, path_pf: Optional[str], path_sf: Optional[str], sep: Optional[str]=None, encoding: Optional[str]=None) -> Dict[str,int]:
    """
    Lecture des CSV avec pandas puis import via import_products_from_dataframes.
    Par défaut, le séparateur est détecté (sep=None) avec l'engine Python.
    """
    if pd is None:
        raise RuntimeError("pandas n'est pas disponible dans cet environnement.")
    read_kwargs = {"dtype": str}
    read_kwargs["sep"] = sep if sep is not None else None
    if encoding is not None:
        read_kwargs["encoding"] = encoding
    read_kwargs["engine"] = "python"
    df_pf = pd.read_csv(path_pf, **read_kwargs) if path_pf else None
    df_sf = pd.read_csv(path_sf, **read_kwargs) if path_sf else None
    return import_products_from_dataframes(conn, df_pf, df_sf)

# ==================== BOM operations ==================

def _is_pf(ix: str) -> bool:
    return _normalize_index_value(ix).startswith("PF")

def _q2dec2(q: Union[str, float, int]) -> Decimal:
    d = Decimal(str(q or "0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if d < Decimal("0"):
        raise ValueError("La quantité doit être >= 0")
    return d

def would_create_cycle(conn: sqlite3.Connection, parent: str, child: str) -> bool:
    parent = _normalize_index_value(parent)
    child = _normalize_index_value(child)
    sql = """
    WITH RECURSIVE reach(x) AS (
      SELECT ?
      UNION ALL
      SELECT e.child_index
      FROM bom_edges e
      JOIN reach r ON e.parent_index = r.x
    )
    SELECT 1 FROM reach WHERE x = ? LIMIT 1;
    """
    row = conn.execute(sql, (child, parent)).fetchone()
    return row is not None

def add_edge(conn: sqlite3.Connection, parent: str, child: str, quantity, user: Optional[str]=None) -> None:
    parent = _normalize_index_value(parent)
    child = _normalize_index_value(child)
    if not parent or not child:
        raise ValueError("Parent ou enfant invalide (Index vide).")
    if _is_pf(child):
        raise ValueError("Un produit fini (PF…) ne peut pas être enfant.")
    if not conn.execute("SELECT 1 FROM products WHERE product_index=?;", (parent,)).fetchone():
        raise ValueError(f"Parent inconnu dans products: {parent}")
    if not conn.execute("SELECT 1 FROM products WHERE product_index=?;", (child,)).fetchone():
        raise ValueError(f"Enfant inconnu dans products: {child}")
    if would_create_cycle(conn, parent, child):
        raise ValueError(f"Insertion refusée : créerait un cycle ({parent} → … → {child} → {parent}).")

    q = _q2dec2(quantity)

    # BEFORE avant modification
    before = conn.execute(
        "SELECT * FROM bom_edges WHERE parent_index=? AND child_index=?;",
        (parent, child)
    ).fetchone()

    # UPSERT unique
    conn.execute(
        "INSERT INTO bom_edges(parent_index, child_index, quantity) VALUES (?,?,?) "
        "ON CONFLICT(parent_index, child_index) DO UPDATE SET quantity=excluded.quantity;",
        (parent, child, str(q))
    )

    # AFTER après modification
    after = conn.execute(
        "SELECT * FROM bom_edges WHERE parent_index=? AND child_index=?;",
        (parent, child)
    ).fetchone()

    log_audit(
                conn,
                user=user,
                action="ADD_EDGE",
                table="bom_edges",
                rec_key=f"{parent}->{child}",
                before_obj=(dict(before) if before else None),
                after_obj=(dict(after) if after else None),
                note=None
            )

    conn.commit()



def remove_edge(conn: sqlite3.Connection, parent: str, child: str, user: Optional[str]=None) -> None:
    # 1) Normalisation des clés pour être cohérent (PF001 == pf001)
    parent = _normalize_index_value(parent)
    child  = _normalize_index_value(child)

    # 2) On lit la ligne AVANT suppression (pour l’enregistrer dans le journal)
    before = conn.execute(
        "SELECT * FROM bom_edges WHERE parent_index=? AND child_index=?;",
        (parent, child)
    ).fetchone()

    # 3) On supprime
    cur = conn.execute(
        "DELETE FROM bom_edges WHERE parent_index=? AND child_index=?;",
        (parent, child)
    )

    # 4) Si rien n’a été supprimé (lien déjà inexistant), on ne log pas
    if cur.rowcount == 0 and before is None:
        conn.commit()
        return

    # 5) On écrit dans audit_log : action REMOVE_EDGE, table bom_edges,
    #    clé lisible "PARENT->CHILD", état 'before' (et pas d'after)
    log_audit(
                conn, user,
                action="REMOVE_EDGE",
                table="bom_edges",
                rec_key=f"{parent}->{child}",
                before_obj=(dict(before) if before else None),
                after_obj=None,
                note=None
            )


    # 6) On valide la transaction
    conn.commit()


# ================ Compute BOM (from SQL) ==============
def compute_bom_tree_sql(conn: sqlite3.Connection, root_index: str) -> Dict:
    """
    Calcule un arbre BOM avec PR recalculé récursivement :
      PR(node) = PA(node) + sum( PR(child) * quantity(child->node) )
    'subtree_cost' et 'PR_contrib' sont alignés pour éviter les doubles comptes.
    """
    root = _normalize_index_value(root_index)
    if not root:
        return {"ok": False, "msg": "Index racine invalide.", "root": root_index}

    # ----- utils SQL -----
    def children_of(p: str) -> List[Tuple[str, float]]:
        rows = conn.execute(
            "SELECT child_index, quantity FROM bom_edges WHERE parent_index=? ORDER BY child_index;",
            (p,)
        ).fetchall()
        return [(r["child_index"], float(r["quantity"])) for r in rows]

    def _select_one_float(sql: str, params: tuple) -> Optional[float]:
        try:
            r = conn.execute(sql, params).fetchone()
        except Exception:
            return None
        if not r:
            return None
        # support row objects or tuples
        v = None
        if isinstance(r, (tuple, list)):
            v = r[0]
        elif isinstance(r, dict):
            # cas Row avec clés
            v = next(iter(r.values()))
        else:
            try:
                v = r[0]
            except Exception:
                v = None
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            try:
                return float(str(v).replace(",", "."))
            except Exception:
                return None

    # Essaie plusieurs colonnes possibles pour le PA selon ton schéma
    def get_pa(ix: str) -> float:
        for col in ("pa", "prix_achat", "purchase_price", "Prix d'achat", "PA"):
            # si la colonne n'existe pas, la requête lèvera — on ignore et on essaie la suivante
            val = _select_one_float(f"SELECT {col} FROM products WHERE product_index=?;", (ix,))
            if val is not None:
                return float(val)
        return 0.0

    def exists_in_products(ix: str) -> bool:
        r = conn.execute("SELECT 1 FROM products WHERE product_index=?;", (ix,)).fetchone()
        return r is not None

    # ----- calcul PR récursif -----
    cycles = set()
    memo_pr: Dict[str, float] = {}
    in_progress: set = set()

    def compute_pr(idx: str) -> float:
        idx = _normalize_index_value(idx)
        if idx in memo_pr:
            return memo_pr[idx]
        if idx in in_progress:
            # cycle -> on casse la boucle avec PR=0 (ou une garde que tu veux)
            cycles.add(idx)
            memo_pr[idx] = 0.0
            return 0.0

        in_progress.add(idx)
        total = get_pa(idx)  # PA propre du nœud
        for cidx, q in children_of(idx):
            total += compute_pr(cidx) * float(q)
        in_progress.remove(idx)
        memo_pr[idx] = float(total)
        return memo_pr[idx]

    # ----- construction de l'arbre "statique" : PR est maintenant le PR recalculé -----
    memo_nodes: Dict[Tuple[str, Optional[str]], Dict] = {}

    def build_static(idx: str, parent: Optional[str], q_direct: Optional[float]) -> Dict:
        key = (idx, parent)
        if key in memo_nodes:
            return memo_nodes[key]
        pr_val = compute_pr(idx)  # <-- PR roulé selon la règle
        children = []
        for cidx, q in children_of(idx):
            qd = float(Decimal(str(q)).quantize(Decimal("0.01")))
            children.append(build_static(cidx, idx, qd))
        node = {
            "Index": idx,
            "PR": pr_val,  # PR *par unité du nœud*, incluant ses enfants
            "Quantité_pour_parent": None if parent is None else (0.0 if q_direct is None else float(q_direct)),
            "children": children,
            "missing": (not exists_in_products(idx)),
            "cycle": idx in cycles
        }
        memo_nodes[key] = node
        return node

    root_static = build_static(root, None, None)

    # ----- expansion avec quantités vs racine -----
    def expand(node: Dict, multiplier: float) -> Dict:
        direct = node.get("Quantité_pour_parent")
        qty_total = multiplier if direct is None else multiplier * float(direct)
        pr_unit = float(node.get("PR", 0.0))           # PR roulé (PA + enfants)
        contrib = pr_unit * qty_total                  # contribution totale de CE nœud pour la racine

        # IMPORTANT : pas d'addition des enfants dans subtree_cost, sinon double-compte
        children_expanded = [expand(c, qty_total) for c in node.get("children", [])]

        return {
            "Index": node["Index"],
            "PR": pr_unit,
            "Quantité_pour_parent": node.get("Quantité_pour_parent"),
            "Quantité_totale_pour_root": qty_total,
            "PR_contrib": contrib,
            "subtree_cost": contrib,   # <-- pas de + sum(enfants)
            "children": children_expanded,
            "missing": node.get("missing", False),
            "cycle": node.get("cycle", False),
        }

    expanded = expand(root_static, 1.0)

    # ----- agrégats à plat (si tu en as besoin) -----
    from collections import defaultdict
    totals = defaultdict(lambda: {"Index": None, "PR": 0.0, "total_qty": 0.0, "total_cost": 0.0})

    def acc(n: Dict):
        ix = n["Index"]
        d = totals[ix]
        d["Index"] = ix
        d["PR"] = n.get("PR", 0.0)
        d["total_qty"] += float(n.get("Quantité_totale_pour_root", 0.0) or 0.0)
        d["total_cost"] += float(n.get("PR_contrib", 0.0) or 0.0)
        for c in n.get("children", []):
            acc(c)

    acc(expanded)
    totals_list = list(totals.values())

    def flatten(n: Dict, parent=None, out=None):
        if out is None:
            out = []
        out.append({
            "Index": n["Index"], "parent": parent, "PR": n["PR"],
            "Quantité_pour_parent": n["Quantité_pour_parent"],
            "Quantité_totale_pour_root": n["Quantité_totale_pour_root"],
            "PR_contrib": n["PR_contrib"], "subtree_cost": n["subtree_cost"],
            "missing": n["missing"], "cycle": n["cycle"]
        })
        for c in n["children"]:
            flatten(c, n["Index"], out)
        return out

    flat_list = flatten(expanded)

    return {
        "ok": True, "root": root, "tree": expanded,
        "totals": totals_list, "flat": flat_list,
        "missing_indices": [], "cycles": sorted(cycles)
    }

# ==================== Simple CLI ======================


def _print(msg: str) -> None:
    print(msg, flush=True)

def main_cli():
    import argparse
    p = argparse.ArgumentParser(description="Reconstruire le schéma et importer les attributs produits (sans ArbreParenté/Quantité:pour/Visible).")
    p.add_argument("--db", default="bom.sqlite3", help="Chemin du fichier SQLite")
    p.add_argument("--pf", help="CSV produits finis (toutes colonnes sauf ArbreParenté)")
    p.add_argument("--sf", help="CSV produits semi-finis (toutes colonnes sauf ArbreParenté/Visible/Quantité:pour)")
    p.add_argument("--sep", default=None, help="Séparateur CSV (détection auto par défaut)")
    p.add_argument("--encoding", default=None, help="Encodage CSV (ex: 'utf-8' ou 'latin-1')")
    args = p.parse_args()

    conn = get_conn(args.db)
    init_schema(conn)
    _print("Schéma initialisé/ajusté.")

    if args.pf or args.sf:
        stats = import_products_from_csv_paths(conn, args.pf, args.sf, sep=args.sep, encoding=args.encoding)
        _print(f"Import terminé: PF={stats.get('pf_rows',0)} lignes, SF={stats.get('sf_rows',0)} lignes.")
    else:
        _print("Aucun CSV fourni; schéma seulement.")

if __name__ == "__main__":
    main_cli()


def upsert_fournisseur(conn, row: dict, user: Optional[str] = None, *, audit: bool = True):
    """
    row = {"nom":..., "categorie":..., "adresse":..., "code_postal":..., "ville":..., "pays":...}
    """
    payload = {k: None for k in ("nom","categorie","adresse","code_postal","ville","pays")}
    payload.update({k: (row.get(k) if k in row else None) for k in payload})

    if not payload.get("nom") or not str(payload["nom"]).strip():
        raise ValueError("Le champ 'nom' est obligatoire pour l'UPSERT fournisseur.")

    # état avant
    before_row = conn.execute(
        "SELECT nom, categorie, adresse, code_postal, ville, pays FROM fournisseurs WHERE nom=?;",
        (payload["nom"],)
    ).fetchone()
    before_obj = dict(before_row) if before_row else None

    # UPSERT
    conn.execute("""
        INSERT INTO fournisseurs (nom, categorie, adresse, code_postal, ville, pays)
        VALUES (:nom, :categorie, :adresse, :code_postal, :ville, :pays)
        ON CONFLICT(nom) DO UPDATE SET
            categorie   = COALESCE(excluded.categorie,   fournisseurs.categorie),
            adresse     = COALESCE(excluded.adresse,     fournisseurs.adresse),
            code_postal = COALESCE(excluded.code_postal, fournisseurs.code_postal),
            ville       = COALESCE(excluded.ville,       fournisseurs.ville),
            pays        = COALESCE(excluded.pays,        fournisseurs.pays);
    """, payload)

    # état après
    after_row = conn.execute(
        "SELECT nom, categorie, adresse, code_postal, ville, pays FROM fournisseurs WHERE nom=?;",
        (payload["nom"],)
    ).fetchone()
    after_obj = dict(after_row) if after_row else None

    # Audit
    if audit:
        if before_obj is None and after_obj is not None:
            # création
            log_audit(conn, user=user, action="CREATE_ROW", table="fournisseurs",
                      rec_key=payload["nom"], before_obj=None, after_obj=after_obj)
        elif before_obj is not None and after_obj is not None:
            # update champ par champ
            for field in ("categorie","adresse","code_postal","ville","pays"):
                old = (before_obj.get(field) if before_obj else None)
                new = (after_obj.get(field)  if after_obj  else None)
                if (old or "") != (new or ""):
                    log_audit(conn, user=user, action="UPDATE_FIELD", table="fournisseurs",
                              rec_key=payload["nom"], field=field,
                              old_value=str(old) if old is not None else None,
                              new_value=str(new) if new is not None else None)
            # et un snapshot de ligne si au moins une diff
            if any((before_obj.get(f) or "") != (after_obj.get(f) or "") for f in ("categorie","adresse","code_postal","ville","pays")):
                log_audit(conn, user=user, action="UPDATE_ROW", table="fournisseurs",
                          rec_key=payload["nom"], before_obj=before_obj, after_obj=after_obj)

    conn.commit()


def upsert_fournisseur(conn, row: dict, user: Optional[str] = None, *, audit: bool = True):
    """
    row = {"nom":..., "categorie":..., "adresse":..., "code_postal":..., "ville":..., "pays":...}
    """
    payload = {k: None for k in ("nom","categorie","adresse","code_postal","ville","pays")}
    payload.update({k: (row.get(k) if k in row else None) for k in payload})

    if not payload.get("nom") or not str(payload["nom"]).strip():
        raise ValueError("Le champ 'nom' est obligatoire pour l'UPSERT fournisseur.")

    # état avant
    before_row = conn.execute(
        "SELECT nom, categorie, adresse, code_postal, ville, pays FROM fournisseurs WHERE nom=?;",
        (payload["nom"],)
    ).fetchone()
    before_obj = dict(before_row) if before_row else None

    # UPSERT
    conn.execute("""
        INSERT INTO fournisseurs (nom, categorie, adresse, code_postal, ville, pays)
        VALUES (:nom, :categorie, :adresse, :code_postal, :ville, :pays)
        ON CONFLICT(nom) DO UPDATE SET
            categorie   = COALESCE(excluded.categorie,   fournisseurs.categorie),
            adresse     = COALESCE(excluded.adresse,     fournisseurs.adresse),
            code_postal = COALESCE(excluded.code_postal, fournisseurs.code_postal),
            ville       = COALESCE(excluded.ville,       fournisseurs.ville),
            pays        = COALESCE(excluded.pays,        fournisseurs.pays);
    """, payload)

    # état après
    after_row = conn.execute(
        "SELECT nom, categorie, adresse, code_postal, ville, pays FROM fournisseurs WHERE nom=?;",
        (payload["nom"],)
    ).fetchone()
    after_obj = dict(after_row) if after_row else None

    # Audit
    if audit:
        if before_obj is None and after_obj is not None:
            # création
            log_audit(conn, user=user, action="CREATE_ROW", table="fournisseurs",
                      rec_key=payload["nom"], before_obj=None, after_obj=after_obj)
        elif before_obj is not None and after_obj is not None:
            # update champ par champ
            for field in ("categorie","adresse","code_postal","ville","pays"):
                old = (before_obj.get(field) if before_obj else None)
                new = (after_obj.get(field)  if after_obj  else None)
                if (old or "") != (new or ""):
                    log_audit(conn, user=user, action="UPDATE_FIELD", table="fournisseurs",
                              rec_key=payload["nom"], field=field,
                              old_value=str(old) if old is not None else None,
                              new_value=str(new) if new is not None else None)
            # et un snapshot de ligne si au moins une diff
            if any((before_obj.get(f) or "") != (after_obj.get(f) or "") for f in ("categorie","adresse","code_postal","ville","pays")):
                log_audit(conn, user=user, action="UPDATE_ROW", table="fournisseurs",
                          rec_key=payload["nom"], before_obj=before_obj, after_obj=after_obj)

    conn.commit()


def delete_fournisseur(conn, nom: str, user: Optional[str] = None, *, audit: bool = True) -> tuple[bool, str]:
    nom = str(nom or "").strip()
    if not nom:
        return False, "Nom fournisseur manquant."

    row = conn.execute(
        "SELECT nom, categorie, adresse, code_postal, ville, pays FROM fournisseurs WHERE nom=?;",
        (nom,)
    ).fetchone()
    if row is None:
        return False, f"Fournisseur introuvable: {nom}"

    before_obj = dict(row)

    conn.execute("DELETE FROM fournisseurs WHERE nom=?;", (nom,))
    if audit:
        log_audit(conn, user=user, action="DELETE_ROW", table="fournisseurs",
                  rec_key=nom, before_obj=before_obj, after_obj=None)
    conn.commit()
    return True, f"Fournisseur supprimé: {nom}"



def import_fournisseurs_from_csv(conn, csv_path: str, encoding: str = "utf-8", sep: str = ",") -> int:
    """
    Lit un CSV style data/Fournisseur.csv et alimente la table fournisseurs.
    Colonnes attendues: Nom, Catégorie, Adresse, Code Postal, Ville, Pays
    """
    df = pd.read_csv(csv_path, encoding=encoding, sep=sep)
    # normalisation colonnes
    rename = {
        "Nom":"nom",
        "Catégorie":"categorie",
        "Adresse":"adresse",
        "Code Postal":"code_postal",
        "Ville":"ville",
        "Pays":"pays",
    }
    df = df.rename(columns=rename)
    ok_cols = ["nom","categorie","adresse","code_postal","ville","pays"]
    for c in ok_cols:
        if c not in df.columns:
            df[c] = None
    n=0
    for _, r in df.fillna("").iterrows():
        upsert_fournisseur(conn, {k: str(r.get(k) or "").strip() for k in ok_cols})
        n+=1
    return n

def get_all_fournisseurs_df(conn) -> pd.DataFrame:
    df = pd.read_sql_query("""
        SELECT nom, categorie, adresse, code_postal, ville, pays
        FROM fournisseurs
        ORDER BY nom COLLATE NOCASE;
    """, conn)
    return df

def get_fournisseur_by_nom(conn, nom: str) -> dict | None:
    cur = conn.execute("""
        SELECT nom, categorie, adresse, code_postal, ville, pays
        FROM fournisseurs WHERE nom = ?;
    """, (nom,))
    row = cur.fetchone()
    return dict(zip([c[0] for c in cur.description], row)) if row else None

def ensure_fournisseurs_seeded_from_csv_if_empty(conn, csv_path: str = "data/Fournisseur.csv"):
    """Pour transition douce : si la table est vide mais le CSV existe, on l’importe une seule fois."""
    cur = conn.execute("SELECT COUNT(*) FROM fournisseurs;")
    count = cur.fetchone()[0]
    if count == 0 and os.path.isfile(csv_path):
        import_fournisseurs_from_csv(conn, csv_path)
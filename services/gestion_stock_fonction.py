import os 
import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from typing import Optional
import streamlit as st
import pandas as pd
from typing import Optional
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# === Intégrations avec l'existant ============================================
from SQL.sql_bom import( 
     log_audit
)
#=====================Tracabillité===================

def get_logged_user() -> str:
    ss = st.session_state
    for k in ("current_user", "user", "username", "login", "email"):
        v = ss.get(k)
        if isinstance(v, dict):
            for kk in ("login", "username", "email", "name"):
                if v.get(kk): return str(v[kk])
        if v: return str(v)
    return os.environ.get("USER") or os.environ.get("USERNAME") or "—"


# --- Guard d'auth : bloque l'accès si non connecté ---
if not st.session_state.get("authenticated", False):
    st.switch_page("Connexion.py")
    st.stop()
    

# === Schéma des nouvelles tables ============================================

CREATE_STOCK_LEVELS = """
CREATE TABLE IF NOT EXISTS stock_levels (
  product_index TEXT PRIMARY KEY REFERENCES products(product_index) ON DELETE CASCADE,
  stock_real INTEGER NOT NULL DEFAULT 0 CHECK (stock_real >= 0),
  real_updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  stock_virtual INTEGER NOT NULL DEFAULT 0 CHECK (stock_virtual >= 0),
  virtual_updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

CREATE_STOCK_HISTORY = """
CREATE TABLE IF NOT EXISTS stock_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL DEFAULT (datetime('now')),
  product_index TEXT NOT NULL REFERENCES products(product_index) ON DELETE CASCADE,
  stock_real INTEGER NOT NULL,
  stock_virtual INTEGER NOT NULL,
  action TEXT NOT NULL CHECK (action IN ('init','edit_real','edit_virtual','manual_edit','receive')),
  user TEXT,
  note TEXT
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS ix_stock_history_idx_ts ON stock_history(product_index, ts);
"""

# Triggers: mise à jour des timestamps quand les colonnes changent
TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS trg_stock_levels_update_real AFTER UPDATE OF stock_real ON stock_levels
    BEGIN
      UPDATE stock_levels
         SET real_updated_at = datetime('now'),
             updated_at = datetime('now')
       WHERE product_index = NEW.product_index;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_stock_levels_update_virtual AFTER UPDATE OF stock_virtual ON stock_levels
    BEGIN
      UPDATE stock_levels
         SET virtual_updated_at = datetime('now'),
             updated_at = datetime('now')
       WHERE product_index = NEW.product_index;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_stock_levels_insert AFTER INSERT ON stock_levels
    BEGIN
      UPDATE stock_levels
         SET updated_at = datetime('now')
       WHERE product_index = NEW.product_index;
    END;
    """
]

def ensure_stock_schema(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(CREATE_STOCK_LEVELS)
        conn.execute(CREATE_STOCK_HISTORY)
        conn.execute(CREATE_INDEXES)
        for t in TRIGGERS:
            conn.execute(t)

# === Accès / opérations ======================================================

@dataclass
class StockRow:
    product_index: str
    stock_real: int
    stock_virtual: int

def _get_stock(conn: sqlite3.Connection, product_index: str) -> Optional[StockRow]:
    row = conn.execute("""
        SELECT product_index, stock_real, stock_virtual
          FROM stock_levels WHERE product_index = ?
    """, (product_index,)).fetchone()
    if not row:
        return None
    return StockRow(*row)

def _init_stock_if_missing(conn: sqlite3.Connection, product_index: str, user: Optional[str]) -> StockRow:
    sr = _get_stock(conn, product_index)
    if sr:
        return sr
    with conn:
        conn.execute("""
            INSERT OR IGNORE INTO stock_levels(product_index, stock_real, stock_virtual)
            VALUES (?, 0, 0)
        """, (product_index,))
        conn.execute("""
            INSERT INTO stock_history(product_index, stock_real, stock_virtual, action, user, note)
            VALUES (?, 0, 0, 'init', ?, 'Initialisation automatique')
        """, (product_index, user))
    return StockRow(product_index, 0, 0)

def _update_stock(conn: sqlite3.Connection, product_index: str,
                  new_real: Optional[int], new_virtual: Optional[int],
                  user: Optional[str], note: Optional[str], action: str) -> Tuple[int, int]:
    before = _get_stock(conn, product_index)
    if not before:
        before = _init_stock_if_missing(conn, product_index, user)

    real = before.stock_real if new_real is None else int(new_real)
    virt = before.stock_virtual if new_virtual is None else int(new_virtual)
    if real < 0 or virt < 0:
        raise ValueError("Les stocks ne peuvent pas être négatifs.")

    with conn:
        # --- mise à jour du stock réel --------------------------------------
        if real != before.stock_real:
            conn.execute(
                "UPDATE stock_levels SET stock_real = ? WHERE product_index = ?",
                (real, product_index)
            )
            # audit cellule (et optionnellement before/after JSON pour la ligne)
            log_audit(
                conn, user, "update", "stock_levels", product_index,
                field="stock_real",
                old_value=before.stock_real,
                new_value=real,
                note=note,
                before_obj={"stock_real": before.stock_real, "stock_virtual": before.stock_virtual},
                after_obj={"stock_real": real, "stock_virtual": virt},
            )

        # --- mise à jour du stock virtuel -----------------------------------
        if virt != before.stock_virtual:
            conn.execute(
                "UPDATE stock_levels SET stock_virtual = ? WHERE product_index = ?",
                (virt, product_index)
            )
            log_audit(
                conn, user, "update", "stock_levels", product_index,
                field="stock_virtual",
                old_value=before.stock_virtual,
                new_value=virt,
                note=note,
                before_obj={"stock_real": before.stock_real, "stock_virtual": before.stock_virtual},
                after_obj={"stock_real": real, "stock_virtual": virt},
            )

        # --- snapshot time-series (après les mises à jour) -------------------
        conn.execute("""
            INSERT INTO stock_history(product_index, stock_real, stock_virtual, action, user, note)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (product_index, real, virt, action, user, note))

    return real, virt


def receive_quantity(conn: sqlite3.Connection, product_index: str, qty: int, user: Optional[str]) -> Tuple[int, int]:
    if qty <= 0:
        raise ValueError("La quantité réceptionnée doit être > 0.")
    current = _init_stock_if_missing(conn, product_index, user)
    # On soustrait du virtuel (pas moins de 0), on ajoute au réel la quantité reçue
    reduce = min(qty, current.stock_virtual)
    new_virtual = max(0, current.stock_virtual - reduce)
    new_real = current.stock_real + reduce
    note = f"Réception {reduce} unités (saisie: {qty})"
    return _update_stock(conn, product_index, new_real, new_virtual, user, note, action="receive")

# === Requêtes d’affichage ====================================================

def load_grid_df(conn: sqlite3.Connection, kind: str) -> pd.DataFrame:
    # Articles suivis (présents dans stock_levels), qu’ils aient du stock ou non
    df = pd.read_sql_query("""
        SELECT p.product_index AS "Index",
               p.libelle_produit AS "Libellé",
               p.couleur AS "Couleur",
               COALESCE(s.stock_real, 0) AS "Stock réel",
               COALESCE(s.stock_virtual, 0) AS "Stock virtuel"
          FROM products p
          JOIN stock_levels s ON s.product_index = p.product_index
         WHERE p.kind = ?
         ORDER BY p.product_index
    """, conn, params=(kind,))
    return df


def load_addable_products(conn: sqlite3.Connection, kind: str) -> pd.DataFrame:
    # Produits de ce type qui ne sont PAS encore dans stock_levels
    return pd.read_sql_query("""
        SELECT p.product_index, p.libelle_produit, p.couleur
          FROM products p
          LEFT JOIN stock_levels s ON s.product_index = p.product_index
         WHERE p.kind = ?
           AND s.product_index IS NULL
         ORDER BY p.libelle_produit, p.product_index
    """, conn, params=(kind,))

def load_candidates_virtual(conn: sqlite3.Connection) -> pd.DataFrame:
    # Produits avec stock virtuel > 0
    return pd.read_sql_query("""
        SELECT p.product_index, p.libelle_produit, p.couleur,
               s.stock_virtual, s.stock_real
          FROM stock_levels s
          JOIN products p ON p.product_index = s.product_index
         WHERE s.stock_virtual > 0
         ORDER BY p.product_index
    """, conn)

def load_history(conn: sqlite3.Connection, product_index: str) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT ts, stock_real, stock_virtual, action, user, note
          FROM stock_history
         WHERE product_index = ?
         ORDER BY ts
    """, conn, params=(product_index,))



# === Intégrations avec l'existant ============================================
from SQL.sql_bom import (
    get_conn,
    log_audit,
    init_audit_schema
)


from services.gestion_stock_fonction import (
    get_logged_user,
    ensure_stock_schema,
    _get_stock,
    _init_stock_if_missing,
    _update_stock,
    receive_quantity,
    load_grid_df,
    load_addable_products,
    load_candidates_virtual,
    load_history
)

# ---------------------- CONFIG SCOPING ---------------------------------------
GS_SCOPE_ID = "gs-root"  # ← id unique qui scopers TOUT le CSS/HTML de cette page

def _inject_scoped_css(scope_id: str) -> None:
    """
    Injecte un CSS scoppé sous #<scope_id> pour éviter tout effet global.
    Renommage des classes génériques: .notif -> .gs-notif, .badge -> .gs-badge.
    """
    css = f"""
<style>
#{scope_id} .gs-notif {{
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    padding: 12px 16px;
    background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
    border: 1px solid #eee;
    display: flex; align-items: center; gap: 8px;
    position: relative;
}}
#{scope_id} .gs-notif::before {{
    content: "•";
    color: #22c55e;
    font-size: 24px; line-height: 0;
}}
#{scope_id} .gs-notif .gs-badge {{
    position: absolute; top: -10px; right: -10px;
    background: #1f2937; color: #fff; font-size: 12px;
    padding: 4px 8px; border-radius: 999px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.12);
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)

# ---------------------- ÉTATS PAR DÉFAUT -------------------------------------
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = None
if "prev_selected_index" not in st.session_state:
    st.session_state["prev_selected_index"] = None
if "expand_audit" not in st.session_state:
    st.session_state["expand_audit"] = False
if "last_kind" not in st.session_state:
    st.session_state["last_kind"] = None

# ---------------------- COMPOSANTS UI ----------------------------------------
def phone_like_banner(text: str, badge: Optional[str] = None, scope_id: str = "gs-banner"):
    st.markdown(
        f"""
        <div id="{scope_id}">
          <style>
            /* CSS scoppé: s'applique uniquement aux éléments rendus DANS ce bloc */
            #{scope_id} .gs-notif {{
               border-radius: 16px;
               box-shadow: 0 6px 18px rgba(0,0,0,0.08);
               padding: 12px 16px;
               background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
               border: 1px solid #eee;
               display: flex; align-items: center; gap: 8px;
               position: relative;
            }}
            #{scope_id} .gs-notif::before {{
               content: "•";
               color: #22c55e;
               font-size: 24px; line-height: 0;
            }}
            #{scope_id} .gs-notif .gs-badge {{
               position: absolute; top: -10px; right: -10px;
               background: #1f2937; color: #fff; font-size: 12px;
               padding: 4px 8px; border-radius: 999px;
               box-shadow: 0 4px 10px rgba(0,0,0,0.12);
            }}
          </style>

          <div class="gs-notif">
            <div style="font-weight:600">{text}</div>
            {f'<div class="gs-badge">{badge}</div>' if badge else ''}
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

#-------------------------FONCTION GRAPH-------------------------
def empty_history_chart(title: str, message: str) -> go.Figure:
    fig = go.Figure()

    # Traces "fantômes" pour conserver la légende et les couleurs
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines+markers",
        name="stock_real", line_shape="hv", hoverinfo="skip", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines+markers",
        name="stock_virtual", line_shape="hv", hoverinfo="skip", showlegend=True
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date/heure",
        yaxis_title="Stock",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=447,
        margin=dict(t=60, r=20, b=40, l=60)
    )

    # Message centré dans le plot
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14)
    )
    return fig

# ---------------------------- PAGE ------------------------------
def render_gestion_stocks(embedded: bool = False, scope_id: str = GS_SCOPE_ID) -> None:
    """
    Rendu principal.
    - embedded=True : pour l'inclure DANS un st.tab d'Application.py
        * pas de use_base_ui
        * pas de set_page_config
        * pas de st.sidebar / switch_page
        * CSS scoppé sous #scope_id
    - embedded=False : exécution autonome comme page
        * on peut appeler use_base_ui
        * CSS scoppé quand même (safe par défaut)
    """
    # 1) Enveloppe racine pour scoper tout le HTML
    st.markdown(f'<div id="{scope_id}">', unsafe_allow_html=True)

    # 2) CSS scoped
    _inject_scoped_css(scope_id)


    # 4) Connexion & schémas
    conn = get_conn()
    init_audit_schema(conn)
    ensure_stock_schema(conn)

    user = get_logged_user() or st.session_state.get("username") or "user"

    # --- État par défaut robuste ---
    ss = st.session_state
    if "kind" not in ss:
        ss["kind"] = "PF"  # Produit fini par défaut au démarrage
    if "last_kind" not in ss:
        ss["last_kind"] = ss["kind"]
    ss.setdefault("selected_index", None)
    ss.setdefault("prev_selected_index", None)
    ss.setdefault("expand_audit", False)

    with st.container():
        # ----------------- Onglet Communication (notification) -------------------
        candidates = load_candidates_virtual(conn)

        if len(candidates) > 0:
            with st.container(border=True):
                    
                phone_like_banner(
                    "Des commandes en attente : des articles ont un stock virtuel > 0. "
                    "Avez-vous réceptionné une partie ou la totalité ?",
                    badge=f"{len(candidates)}"
                )
                with st.container():
                    left, right = st.columns([2, 1])
                    with left:
                        sel_rcv = st.selectbox(
                            "Article à réceptionner",
                            options=[f"{r['product_index']} — {r['libelle_produit']}" for _, r in candidates.iterrows()],
                            index=0,
                            key=f"{scope_id}_sel_rcv"
                        )
                        selected_index_rcv = sel_rcv.split(" — ")[0]
                        current_row = candidates[candidates["product_index"] == selected_index_rcv].iloc[0]
                        st.caption(f"Virtuel actuel: **{int(current_row['stock_virtual'])}** · Réel actuel: **{int(current_row['stock_real'])}**")

                    with right:
                        qty = st.number_input("Quantité réceptionnée", min_value=0, step=1, value=0, key=f"{scope_id}_qty_in")
                        if st.button("✅ Valider la réception", key=f"{scope_id}_btn_rcv"):
                            try:
                                new_real, new_virtual = receive_quantity(conn, selected_index_rcv, int(qty), user)
                                st.success(f"Réception enregistrée. Réel: {new_real} · Virtuel: {new_virtual}")
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

    # --------------------- Trois colonnes principales -----------------------

    col1_header,col2_header = st.columns([5,3])
    col1, col2 = st.columns([5,3])
    col1footer,col2footer,col3footer,col4footer=st.columns(4)
    # Radio PF/SF contrôle le tableau à gauche et les opérations
    
    

    

 
    with col2_header:

        col1h,col2h,col3h = st.columns(3)

        with col2h :

            st.markdown("")
            st.markdown("")
            kind = st.radio(
                "",
                options=["PF", "SF"],
                index=0,
                horizontal=True,
                key="kind"  # state partagé volontairement (PF/SF global à l’app)
            )

            if st.session_state["last_kind"] != st.session_state["kind"]:
                st.session_state["prev_selected_index"] = None
                st.session_state["selected_index"] = None
                st.session_state["expand_audit"] = False
                st.session_state["last_kind"] = st.session_state["kind"]

                kind = st.session_state["kind"]

    with col1:
      
        with st.container(border=True):

            df_grid = load_grid_df(conn, kind=kind)

            if df_grid.empty:

                st.info("Aucun article suivi pour ce type. Utilisez le sélecteur ci-dessus pour en ajouter.")
                selected_index = None
            else:
                # --- 1) Colonne checkbox dédiée au tout début ----------------------------
                # On insère une colonne factice "☑️" qui recevra les checkboxes AgGrid
                if "☑️" not in df_grid.columns:
                    df_grid.insert(0, "☑️", "")  # valeur factice, non utilisée par AgGrid

                gb = GridOptionsBuilder.from_dataframe(df_grid)

                # --- 2) Sélection au clic sur checkbox (une seule) ------------------------
                gb.configure_selection(selection_mode="single", use_checkbox=True)

                # --- 3) Options de grille -------------------------------------------------
                gb.configure_grid_options(
                    domLayout='normal',
                    suppressRowClickSelection=True,  # évite de sélectionner en cliquant la ligne entière
                    rowSelection='single',
                    rowDeselection=True
                )

                # --- 4) Par défaut: colonnes flex pour occuper toute la largeur -----------

                gb.configure_default_column(
                    minWidth=100,
                    resizable=True,
                    flex=1,             # <- remplit l'espace disponible
                    filter=True,        # filtres actifs partout
                    sortable=True
                )

                # --- 5) Colonne checkbox figée à 40 px et sans filtres/menus -------------
                gb.configure_column(
                    "☑️",
                    headerCheckboxSelection=True,   
                    checkboxSelection=True,
                    pinned="left",
                    editable=False,
                    suppressMovable=True,

                    width=40, minWidth=40, maxWidth=40,
                    resizable=False,
                    suppressSizeToFit=True,         

                    sortable=False,
                    filter=False,
                    suppressMenu=True,              
                    suppressHeaderMenuButton=True,  
                    suppressFloatingFilter=True
                )

                # --- 6) Ta mise en forme spécifique --------------------------------------
                gb.configure_column(
                    "Stock réel",
                    sort="des",
                    sortIndex=0,
                    #cellStyle={"backgroundColor": "#427DCA"},
                    headerClass="stock-real-header"
                )
                gb.configure_column(
                    "Stock virtuel",
                    sort="des",
                    sortIndex=1,
                    #cellStyle={"backgroundColor": "#A3D5EC"},
                    headerClass="stock-virtual-header"
                )

                grid_options = gb.build()

                grid_resp = AgGrid(
                    df_grid,
                    gridOptions=grid_options,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    height=500,
                    key=f"{scope_id}_grid_main_{kind}",

                    # IMPORTANT : on laisse les colonnes flex gérer la largeur
                    fit_columns_on_grid_load=True,
                    allow_unsafe_jscode=True
                )

                # --- 7) Récup sélection (DataFrame -> list[dict]) -------------------------
                sel = grid_resp.get("selected_rows", []) if isinstance(grid_resp, dict) else getattr(grid_resp, "selected_rows", [])
                if isinstance(sel, pd.DataFrame):
                    sel = sel.to_dict("records")
                elif sel is None:
                    sel = []


                # Récup sélection (DataFrame -> list[dict])
                sel = grid_resp.get("selected_rows", []) if isinstance(grid_resp, dict) else getattr(grid_resp, "selected_rows", [])
                if isinstance(sel, pd.DataFrame):
                    sel = sel.to_dict("records")
                elif sel is None:
                    sel = []

        prev = st.session_state["selected_index"]
        new_sel = sel[0].get("Index") if len(sel) > 0 else None

        if new_sel != prev:
            st.session_state["prev_selected_index"] = prev
            st.session_state["selected_index"] = new_sel
            st.session_state["expand_audit"] = bool(new_sel)

        selected_index = st.session_state["selected_index"]

    

    if selected_index:

        # Charger valeur actuelle
        sr = _get_stock(conn, selected_index) or _init_stock_if_missing(conn, selected_index, user)
        st.caption(f"Stock de l'Article **{selected_index}**")
        with col1footer :
            real_val = st.number_input("Stock réel", min_value=0, step=1, value=int(sr.stock_real), key=f"{scope_id}_edit_real")
        with col2footer:
            virt_val = st.number_input("Stock virtuel", min_value=0, step=1, value=int(sr.stock_virtual), key=f"{scope_id}_edit_virtual")
        with col3footer:
            note = st.text_input("Note (optionnelle)", placeholder="Raison de la modification…", key=f"{scope_id}_edit_note")
        with col4footer:
            st.markdown("")
            st.markdown("")
            if st.button("💾 Enregistrer", key=f"{scope_id}_btn_save_edit",use_container_width=True):
                try:
                    nr, nv = _update_stock(conn, selected_index, int(real_val), int(virt_val), user, note, action="manual_edit")
                    st.success(f"Modifications enregistrées. Réel: {nr} · Virtuel: {nv}")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    else:
        st.caption("Aucun article sélectionné")
        with col1footer:
            st.number_input("Stock réel", min_value=0, step=1, value=0, key=f"{scope_id}_edit_real_disabled", disabled=True)
        with col2footer:
            st.number_input("Stock virtuel", min_value=0, step=1, value=0, key=f"{scope_id}_edit_virtual_disabled", disabled=True)
        with col3footer:
            st.text_input("Note (optionnelle)", placeholder="Raison de la modification…", key=f"{scope_id}_edit_note_disabled", disabled=True)
        with col4footer:
            st.markdown("")
            st.markdown("")
            st.button("💾 Enregistrer", key=f"{scope_id}_btn_save_edit_disabled", use_container_width=True, disabled=True)

    # Bloc édition + valeurs actuelles
   
    with col1_header:
        
        with st.container(border=True):
                
            # ➕ Ajouter un article au suivi (piloté par le st.radio PF/SF)
            addables = load_addable_products(conn, kind)

            if addables.empty:
                st.caption(f"Tous les articles de type {kind} sont déjà suivis.")
            else:
                options = [f"{row.product_index} — {row.libelle_produit}" for _, row in addables.iterrows()]
                choice = st.selectbox(
                    "➕ Ajouter un article au suivi",
                    options=options,
                    key=f"{scope_id}_add_select"
                )
                add_l, add_r = st.columns([1, 1])
                with add_l:
                    if st.button("Ajouter à la liste", key=f"{scope_id}_btn_add_stockrow"):
                        try:
                            idx = choice.split(" — ")[0]
                            _init_stock_if_missing(conn, idx, user)  # crée stock_levels(0,0) + history('init')
                            st.success(f"Article {idx} ajouté au suivi (réel=0, virtuel=0).")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                with add_r:
                    st.caption("L’article ajouté apparaîtra dans le tableau ci-dessous.")


    # Graphique d'évolution
    with col2:
        container3 = st.container(border=True)
        with container3:
            st.subheader("Évolution des stocks")
            if selected_index:
                dfh = load_history(conn, selected_index)
                if dfh.empty:
                    # Remplacer st.info par un graphique vide avec message
                    fig = empty_history_chart(
                        title=f"Historique — {selected_index}",
                        message="Pas encore d'historique pour cet article."
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Courbes : stock_real et stock_virtual dans le temps
                    dfm = dfh.melt(
                        id_vars=["ts"],
                        value_vars=["stock_real", "stock_virtual"],
                        var_name="Série",
                        value_name="Stock"
                    ).sort_values("ts")

                    # Lignes en escalier
                    fig = px.line(
                        dfm,
                        x="ts",
                        y="Stock",
                        color="Série",
                        markers=True,
                        title=f"Historique — {selected_index}",
                        line_shape="hv"
                    )

                    fig.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        height=447
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Remplacer st.info par un graphique vide avec message
                fig = empty_history_chart(
                    title="Historique",
                    message="Sélectionnez un article."
                )
                st.plotly_chart(fig, use_container_width=True)

    # --------------------- Menu déroulant : édition de l'audit ---------------
    if selected_index is None:
        pass
    else:
        with st.expander("🛠️ Éditer l’audit de l’article sélectionné", expanded=st.session_state["expand_audit"]):
            dfh = pd.read_sql_query(
                """
                SELECT id, ts, action, user, note, stock_real, stock_virtual
                FROM stock_history
                WHERE product_index = ?
                ORDER BY ts DESC
                """,
                conn,
                params=(selected_index,)
            )

            if dfh.empty:
                st.info("Aucune entrée d’audit pour cet article.")
            else:
                gb2 = GridOptionsBuilder.from_dataframe(dfh)
                gb2.configure_selection(selection_mode="multiple", use_checkbox=True)
                gb2.configure_default_column(editable=True)
                gb2.configure_column("id", editable=False)

                grid2 = AgGrid(
                    dfh,
                    gridOptions=gb2.build(),
                    update_mode=GridUpdateMode.VALUE_CHANGED | GridUpdateMode.SELECTION_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    fit_columns_on_grid_load=True,
                    height=200,
                    key=f"{scope_id}_audit_grid_{selected_index}",
                )

                # --- Alignement propre par id + comparaison colonne par colonne -------------
                cmp_cols = ["ts", "action", "user", "note", "stock_real", "stock_virtual"]

                orig = dfh.set_index("id")
                ed = grid2.data.set_index("id")

                # S'assurer que les colonnes existent et sont dans le même ordre
                ed = ed.reindex(columns=cmp_cols)
                orig = orig.reindex(columns=cmp_cols)

                # On ne compare que les ids communs
                common = orig.index.intersection(ed.index)

                # Normalise les types
                left = orig.loc[common, cmp_cols].astype(str)
                right = ed.loc[common, cmp_cols].astype(str)

                diff_mask = (left != right).any(axis=1)

                # changed_rows = lignes modifiées
                changed_rows = ed.loc[common, :][diff_mask].reset_index()

                colA, colB = st.columns([1, 1])
                with colA:
                    if st.button("💾 Enregistrer les modifications", key=f"{scope_id}_btn_save_audit"):
                        with conn:
                            for _, row in changed_rows.iterrows():
                                before = conn.execute(
                                    "SELECT ts, action, user, note, stock_real, stock_virtual FROM stock_history WHERE id = ?",
                                    (int(row["id"]),)
                                ).fetchone()

                                conn.execute(
                                    """
                                    UPDATE stock_history
                                    SET ts = ?, action = ?, user = ?, note = ?, stock_real = ?, stock_virtual = ?
                                    WHERE id = ?
                                    """,
                                    (
                                        row["ts"], row["action"], row.get("user"), row.get("note"),
                                        int(row["stock_real"]), int(row["stock_virtual"]), int(row["id"])
                                    )
                                )

                                log_audit(
                                    conn, user, "update", "stock_history", str(row["id"]),
                                    before_obj={"ts": before[0], "action": before[1], "user": before[2],
                                                "note": before[3], "stock_real": before[4], "stock_virtual": before[5]},
                                    after_obj={"ts": row["ts"], "action": row["action"], "user": row.get("user"),
                                               "note": row.get("note"), "stock_real": int(row["stock_real"]),
                                               "stock_virtual": int(row["stock_virtual"])},
                                    note="Édition manuelle de l’audit"
                                )

                        st.success("Modifications enregistrées.")
                        st.rerun()
                with colB:
                    # Normaliser la sélection
                    sel_rows = grid2.selected_rows
                    if isinstance(sel_rows, pd.DataFrame):
                        sel_rows = sel_rows.to_dict("records")
                    elif sel_rows is None:
                        sel_rows = []

                    if st.button("🗑️ Supprimer la sélection", key=f"{scope_id}_btn_del_audit"):
                        if len(sel_rows) == 0:
                            st.warning("Sélectionnez au moins une ligne.")
                        else:
                            ids = tuple(int(r["id"]) for r in sel_rows)
                            with conn:
                                # Audit générique (avant suppression)
                                for rid in ids:
                                    b = conn.execute(
                                        "SELECT ts, action, user, note, stock_real, stock_virtual FROM stock_history WHERE id = ?",
                                        (rid,)
                                    ).fetchone()
                                    log_audit(
                                        conn, user, "delete", "stock_history", str(rid),
                                        before_obj={"ts": b[0], "action": b[1], "user": b[2],
                                                    "note": b[3], "stock_real": b[4], "stock_virtual": b[5]},
                                        after_obj=None, note="Suppression audit"
                                    )
                                conn.execute(
                                    f"DELETE FROM stock_history WHERE id IN ({','.join(['?']*len(ids))})",
                                    ids
                                )
                            st.success(f"{len(ids)} ligne(s) supprimée(s).")
                            st.rerun()

    # 5) Ferme le conteneur scopé
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- PAGE (exécution directe) -----------------------------
def main():
    # Mode page autonome (garde use_base_ui)
    render_gestion_stocks(embedded=False, scope_id=GS_SCOPE_ID)
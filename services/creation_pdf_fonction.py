import os,re,time
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pandas as pd
import streamlit as st
import pymupdf as fitz
from PIL import Image, ImageFilter
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode,JsCode
from streamlit_pdf_viewer import pdf_viewer

# ==== Imports applicatifs (ne rien recoder) ====
from services.donnees import (
        Kind, 
        load_products, 
        _normalize_index_value
)

from services.interaction_DB import get_children

from services.gestion_stock_fonction import (
    _init_stock_if_missing, 
    _update_stock
)


from SQL.sql_bom import (
    init_schema, 
    init_audit_schema,
    get_all_fournisseurs_df, 
    ensure_fournisseurs_seeded_from_csv_if_empty,
    upsert_fournisseur, 
    delete_fournisseur
)

from SQL.db import get_conn,DB_PATH
# ───────────────── Constantes ─────────────────
MODE_PAIEMENT = ["Virement", "Chèque"]
DELAIS_PAIEMENT = ["10 Jrs", "30 Jrs", "60 Jrs"]

# ----------------------- Config -----------------------
DEFAULT_DIR = "data/commande"
ROOT_DIR = Path(DEFAULT_DIR)
ROOT_DIR.mkdir(parents=True, exist_ok=True)

BASE_DISPLAY_WIDTH = 700
SCALE = 0.75
DISPLAY_WIDTH = int(BASE_DISPLAY_WIDTH * SCALE)  # largeur fixe de rendu des aperçus
SUPER_SAMPLE = 5  # sur-échantillonnage pour netteté
COMMAND_DIR = "data/commande"

PDF_PAT = re.compile(r"^LGF_(\d{5})__?\s*(.+?)\s*\.pdf$", re.IGNORECASE)
PLACEHOLDER_NAME = "_placeholder_blurred.pdf"
PLACEHOLDER_PATH = str((ROOT_DIR / PLACEHOLDER_NAME).resolve())
BLUR_RADIUS = 6



conn = get_conn()
init_schema(conn)
init_audit_schema(conn) 
ensure_fournisseurs_seeded_from_csv_if_empty(conn, "data/Fournisseur.csv")


def _get_conn_for_suppliers():
    c = get_conn(DB_PATH)
    init_schema(c)
    init_audit_schema(c)
    return c


def load_suppliers_df():
    c = _get_conn_for_suppliers()
    # Schéma détecté dans bom.sqlite3 : table 'fournisseurs'
    df = pd.read_sql_query("""
        SELECT id, nom, categorie, adresse, code_postal, ville, pays
        FROM fournisseurs
        ORDER BY nom COLLATE NOCASE
    """, c)
    return df


def _empty_supplier():
    return {
        "nom": "",
        "categorie": "",
        "adresse": "",
        "code_postal": "",
        "ville": "",
        "pays": "",
    }

@st.dialog("Modifier un fournisseur", width="large")
def dialog_edit_supplier(row: dict):
    st.caption(f"ID interne : {row.get('id')}")
    with st.form("form_edit_supplier", border=True):
        col1, col2 = st.columns(2)
        with col1:
            nom = st.text_input("Nom", value=row.get("nom", ""))
            categorie = st.text_input("Catégorie", value=row.get("categorie", ""))
            code_postal = st.text_input("Code postal", value=row.get("code_postal", ""))
        with col2:
            ville = st.text_input("Ville", value=row.get("ville", ""))
            pays = st.text_input("Pays", value=row.get("pays", ""))
        adresse = st.text_area("Adresse", value=row.get("adresse", ""), height=80)

        c1, c2,c3,c4 = st.columns(4)
        with c2:
            submitted = st.form_submit_button("Enregistrer",use_container_width=True)
        with c3:
            cancel = st.form_submit_button("Fermer",use_container_width=True)

    if cancel:
        st.session_state.sup_dialog_open = None
        st.rerun()

    if submitted:
        payload = {
            "id": int(row["id"]),
            "nom": nom.strip(),
            "categorie": (categorie or "").strip(),
            "adresse": (adresse or "").strip(),
            "code_postal": (code_postal or "").strip(),
            "ville": (ville or "").strip(),
            "pays": (pays or "").strip(),
        }
        with _get_conn_for_suppliers() as c:
            upsert_fournisseur(c, payload, user=st.session_state.get("username", ""))

        st.toast(f"« {payload['nom']} » enregistré", icon="💾")
        st.session_state.sup_dialog_open = None
        time.sleep(1.2)
        st.rerun()

    
@st.dialog("Créer un fournisseur", width="large")
def dialog_create_supplier():
    row = st.session_state.sup_row_cache or _empty_supplier()
    with st.form("form_create_supplier", border=True):
        col1, col2 = st.columns(2)
        with col1:
            nom = st.text_input("Nom", value=row.get("nom", ""))
            categorie = st.text_input("Catégorie", value=row.get("categorie", ""))
            code_postal = st.text_input("Code postal", value=row.get("code_postal", ""))
        with col2:
            ville = st.text_input("Ville", value=row.get("ville", ""))
            pays = st.text_input("Pays", value=row.get("pays", ""))
        adresse = st.text_area("Adresse", value=row.get("adresse", ""), height=80)

        c1, c2,c3,c4 = st.columns([2,1,1,2])
        with c2:
            submitted = st.form_submit_button("Créer",use_container_width=True)
        with c3:
            cancel = st.form_submit_button("Fermer",use_container_width=True)

    if cancel:
        st.session_state.sup_dialog_open = None
        st.rerun()

    if submitted:
        payload = {
            "nom": (nom or "").strip(),
            "categorie": (categorie or "").strip(),
            "adresse": (adresse or "").strip(),
            "code_postal": (code_postal or "").strip(),
            "ville": (ville or "").strip(),
            "pays": (pays or "").strip(),
        }
        with _get_conn_for_suppliers() as c:
            upsert_fournisseur(c, payload, user=st.session_state.get("username", ""))
            new_row = c.execute(
                "SELECT id FROM fournisseurs WHERE nom = ? ORDER BY id DESC LIMIT 1;",
                (payload["nom"],)
            ).fetchone()
            if new_row:
                st.session_state.sup_selected_id = int(new_row[0])

        st.toast(f"« {payload['nom']} » créé", icon="✅")
        time.sleep(1.2)
        st.session_state.sup_dialog_open = None
        st.rerun()



conn = get_conn(DB_PATH)
try:
    ensure_fournisseurs_seeded_from_csv_if_empty(conn, "data/Fournisseur.csv")
except Exception:
    pass


df_fourn = get_all_fournisseurs_df(conn)
Nom_fournisseur = df_fourn["nom"].tolist()

def _invalidate_pdf_cache_and_rescan(select_idx_norm: Optional[str] = None, select_cmd_no: Optional[int] = None):

    st.session_state["cache_buster"] = st.session_state.get("cache_buster", 0) + 1

    try:
        scan_pdf_inventory.clear()
    except Exception:
        pass
    try:
        list_pdfs.clear()
    except Exception:
        pass

    if select_idx_norm is not None:
        _on_selection_change(select_idx_norm)
    if select_cmd_no is not None:
        _on_cmd_no_change(int(select_cmd_no))

    st.rerun()

def _infos_fournisseur(nom: str):
    r = df_fourn.loc[df_fourn["nom"] == nom].iloc[0].to_dict()
    return {
        "ville": r.get("ville",""),
        "adresse": r.get("adresse",""),
        "code_postal": r.get("code_postal",""),
        "pays": r.get("pays",""),
        "categorie": r.get("categorie",""),
    }

def _generate_blurred_pdf(source_pdf: str, out_pdf: str, blur_radius: int = BLUR_RADIUS) -> None:
    """
    Crée un PDF flouté (toutes les pages) à partir d'un PDF source.
    """
    src = fitz.open(source_pdf)
    out = fitz.open()

    for p in src:
        # Rendu image de la page
        # On calcule un zoom pour un rendu suffisamment net avant flou
        page_width_pt = p.rect.width or 595  # fallback A4
        zoom = (BASE_DISPLAY_WIDTH / page_width_pt) if page_width_pt else 1.0
        mat = fitz.Matrix(zoom, zoom)
        pix = p.get_pixmap(matrix=mat, alpha=False)

        # Pixmap -> PIL -> flou -> bytes PNG
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Convertir l'image floutée en page PDF
        # 1) sauver en PNG en mémoire
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # 2) créer une page à la même taille que la page source (en points)
        page = out.new_page(width=p.rect.width, height=p.rect.height)
        rect = fitz.Rect(0, 0, p.rect.width, p.rect.height)
        page.insert_image(rect, stream=buf.read(), keep_proportion=False)

    out.save(out_pdf)
    out.close()
    src.close()



def ensure_placeholder_blurred(
    command_dir: str,
    all_pdfs_sorted: List[Tuple[int, str, str]],
    force: bool = False
) -> Optional[str]:
    try:
        placeholder_path = PLACEHOLDER_PATH

        if force or not os.path.isfile(placeholder_path):
            source_path = None
            for (_num, _idx, p) in all_pdfs_sorted:
                if os.path.isfile(p):
                    source_path = p
                    break

            if source_path:
                _generate_blurred_pdf(source_path, placeholder_path, BLUR_RADIUS)
            else:
                doc = fitz.open()
                page = doc.new_page(width=595, height=842)
                page.insert_textbox(
                    fitz.Rect(36, 36, 559, 806),
                    "Aucun PDF sélectionné.\n(placeholder)",
                    fontsize=18,
                    align=1,
                )
                doc.save(placeholder_path)
                doc.close()

        return placeholder_path
    except Exception as e:
        st.error(f"Impossible de créer le placeholder flouté : {e}")
        return None


def _init_state():
    st.session_state.setdefault("current_kind", Kind.PF)
    st.session_state.setdefault("selected_index_norm", None)
    st.session_state.setdefault("selected_cmd_no", None)
    st.session_state.setdefault("current_page", 1)
    st.session_state.setdefault("confirm_delete", False)
    st.session_state.setdefault("cache_buster", 0)
    
    st.session_state.setdefault("sup_dialog_open", None)   # "edit" | "create" | None
    st.session_state.setdefault("sup_selected_id", None)
    st.session_state.setdefault("sup_row_cache", None)
_init_state()

def _on_kind_change(new_kind: Kind):
    st.session_state["current_kind"] = new_kind
    st.session_state["selected_index_norm"] = None
    st.session_state["selected_cmd_no"] = None
    st.session_state["current_page"] = 1
    st.session_state["confirm_delete"] = False

def _on_selection_change(new_index_norm: Optional[str], raw_index: Optional[str] = None):
    st.session_state["selected_index_norm"] = new_index_norm
    st.session_state["selected_index_raw"] = raw_index  # <- nouveau
    st.session_state["selected_cmd_no"] = None
    st.session_state["current_page"] = 1
    st.session_state["confirm_delete"] = False

def _on_cmd_no_change(new_no: Optional[int]):
    st.session_state["selected_cmd_no"] = new_no
    st.session_state["current_page"] = 1

def as_kind_enum(v) -> Kind:
    """Retourne Kind.PF / Kind.SF quelle que soit la forme d'entrée (Enum, 'PF', 'SF')."""
    if isinstance(v, Kind):
        return v
    s = str(v).upper()
    return Kind.PF if s == "PF" else Kind.SF

def kind_key_tag(v) -> str:
    """Retourne 'PF' ou 'SF' pour composer les clés Streamlit, quelle que soit la forme (Enum/str)."""
    # .name si Enum, .value si Enum avec valeur, sinon string upper
    return getattr(v, "name", None) or getattr(v, "value", None) or str(v).upper()

# ───────────────── Helpers DB/Stock/BOM ─────────────────
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

def render_consultation_grid(df_products, key_prefix: str = "grid"):
    df_view = df_products.copy()
    df_view_display = df_view.drop(columns=["Index_norm"]) if "Index_norm" in df_view.columns else df_view

    gob = GridOptionsBuilder.from_dataframe(df_view_display)
   
    gob.configure_default_column(
        filter=True,
        floatingFilter=False,          # << PAS de 2e rangée
        sortable=False,
        resizable=True,
        suppressMenu=False,            # << Affiche le menu 3 barres
        menuTabs=['filterMenuTab'],    # << Uniquement l’onglet Filtre
    )

    
    gob.configure_selection(selection_mode="single", use_checkbox=True)
    gob.configure_column(
        "__select__",                  # colonne technique
        headerName="☑️",
        checkboxSelection=True,        # cases sur les lignes
        headerCheckboxSelection=False, # PAS de case dans l’en-tête
        width=30,
        pinned="left",
        resizable=False,
        sortable=False,
        filter=False,
        suppressMenu=True,             # pas de menu sur la colonne “☑️”
        lockPosition=True,
        suppressSizeToFit=True,
    )
    # Empêcher toute case sur les colonnes métier
    for col in ["Libellé produit", "Index", "Couleur", "Référence", "Index_norm"]:
        if col in df_view_display.columns:
            gob.configure_column(col, checkboxSelection=False, editable=False)

    gob.configure_grid_options(
        getRowId=JsCode("function(p){ return String(p.data.Index); }"),
        domLayout="normal",
        deltaRowDataMode=True,
        suppressRowClickSelection=True,
        rowSelection="single",
        # Ne jamais mettre de checkbox ailleurs que sur __select__
        isRowSelectable=JsCode("function(){ return true; }")  # ok pour selection, mais pas de checkbox auto
    )


    
    gob.configure_grid_options(
        getRowId=JsCode("function(p){ return String(p.data.Index); }"),
        domLayout="normal",
        deltaRowDataMode=True,
        suppressRowClickSelection=True,   # clic ligne ≠ coche
        rowSelection="single",
    )

    # Pré-sélection visuelle si déjà choisie
    preselect_raw = st.session_state.get("selected_index_raw")
    if preselect_raw is not None:
        gob.configure_grid_options(
            onGridReady=JsCode(f"""
                function(params){{
                  const target = {repr(str(preselect_raw))};
                  let nodeToSelect = null;
                  params.api.forEachNode(function(n){{
                    if(String(n.data.Index) === target) nodeToSelect = n;
                  }});
                  if(nodeToSelect){{
                    nodeToSelect.setSelected(true);
                    params.api.ensureNodeVisible(nodeToSelect);
                  }}
                }}
            """)
        )

    grid_options = gob.build()

    # Sécurité : imposer menu “Filtre” sur toutes les colonnes (sauf ☑️)
    for col in grid_options.get("columnDefs", []):
        if col.get("field") != "__select__":
            col["menuTabs"] = ['filterMenuTab']

    grid_resp = AgGrid(
        df_view_display,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        allow_unsafe_jscode=True,
        height=360,
        key=f"{key_prefix}_aggrid",
    )

    # Gestion sélection
    rows = grid_resp.get("selected_rows", [])
    has_selection = (not rows.empty) if isinstance(rows, pd.DataFrame) else bool(rows)
    if has_selection:
        row = rows.iloc[0].to_dict() if isinstance(rows, pd.DataFrame) else rows[0]
        index_value = row.get("Index")
        index_norm = _normalize_index_value(index_value) if index_value else None
        if index_norm != st.session_state.get("selected_index_norm"):
            _on_selection_change(index_norm, index_value)
    else:
        if st.session_state.get("selected_index_norm") is not None:
            _on_selection_change(None, None)



def render_selection_and_actions(map_index_to_pdfs: Dict[str, List[Tuple[int, str]]], key_prefix: str = "sel"):
    idx = st.session_state.get("selected_index_norm")

    if idx and idx in map_index_to_pdfs:
        nums = [int(num) for (num, _) in map_index_to_pdfs[idx]]  # déjà triés desc

        # 👉 si aucun numéro pour cet index (liste vide), on évite la selectbox
        if not nums:
            st.warning("Aucun PDF disponible pour cet index.")
            return

        default_no = st.session_state.get("selected_cmd_no")
        if default_no not in nums:
            default_no = nums[0]  # le plus récent (liste triée desc)

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        with c2 : 
            st.markdown("")
            st.markdown("")
        with c3 : 
            st.markdown("")
            st.markdown("")
            st.markdown("")
        with c4 : 
            st.markdown("")
            st.markdown("")
            
        with c1:
            # ⚠️ format_func doit renvoyer une string
            chosen_no = st.selectbox(
                "N° de commande",
                options=nums,
                index=nums.index(default_no),
                key=f"{key_prefix}_cmdno",
                format_func=lambda n: str(n),  # <- important
            )
            if chosen_no != st.session_state.get("selected_cmd_no"):
                _on_cmd_no_change(int(chosen_no))

        # Récupérer le chemin pour téléchargement/suppression
        path = None
        sel_no = st.session_state.get("selected_cmd_no")
        if sel_no is not None:
            for (n, p) in map_index_to_pdfs[idx]:
                if int(n) == int(sel_no):
                    path = p
                    break

        with c2:

            if path and os.path.isfile(path):
                with open(path, "rb") as f:
                    st.download_button(
                        "💾",
                        data=f.read(),  # lire en bytes
                        file_name=os.path.basename(path),
                        mime="application/pdf",
                        key=f"{key_prefix}_dl",
                        use_container_width=True,
                    )
            else:

                st.download_button(
                    "💾",
                    data=b"",
                    file_name="inexistant.pdf",
                    disabled=True,
                    key=f"{key_prefix}_dl_dis",
                    use_container_width=True,
                )

        with c3:
            st.checkbox("➡️", key=f"{key_prefix}_confirm")
        with c4:
            can_delete = bool(path) and st.session_state.get(f"{key_prefix}_confirm", False)
            if st.button("🗑️", disabled=not can_delete, key=f"{key_prefix}_delbtn", use_container_width=True):
                try:
                    if os.path.abspath(path) == os.path.abspath(PLACEHOLDER_PATH):
                        st.warning("Le PDF par défaut ne peut pas être supprimé.")
                        return
                    os.remove(path)

                    st.success(f"Supprimé : {os.path.basename(path)}")
                    # Invalidation ciblée
                    st.session_state["cache_buster"] += 1
                    # Re-scan
                    map2, _ = scan_pdf_inventory(COMMAND_DIR, cache_buster=st.session_state["cache_buster"])
                    # Si l'index n'a plus de PDFs → retirer la sélection
                    if st.session_state["selected_index_norm"] not in map2:
                        _on_selection_change(None)
                    else:
                        # Si on a supprimé le dernier n° → prendre le nouveau plus grand
                        remaining = [n for (n, _) in map2[st.session_state["selected_index_norm"]]]
                        _on_cmd_no_change(int(remaining[0]) if remaining else None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Impossible de supprimer : {e}")
    else:
        # Toujours afficher les contrôles mais désactivés
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

        with c2 : 
            st.markdown("")
            st.markdown("")
        with c3 : 
            st.markdown("")
            st.markdown("")
            st.markdown("")
        with c4 : 
            st.markdown("")
            st.markdown("")

        with c1:
            st.selectbox(
                "N° de commande",
                options=[],
                index=None,
                key=f"{key_prefix}_cmdno_disabled",
                placeholder="Sélectionnez un produit d'abord",
                disabled=True,
            )
        with c2:
            st.download_button(
                "💾",
                data=b"",
                file_name="inexistant.pdf",
                disabled=True,
                key=f"{key_prefix}_dl_dis",
                use_container_width=True,
            )
        with c3:
            st.checkbox("➡️", value=False, disabled=True, key=f"{key_prefix}_confirm_dis")
        with c4:
            st.button("🗑️", disabled=True, key=f"{key_prefix}_delbtn_dis", use_container_width=True)
    
    st.subheader("Fournisseurs Base de donnée")
    df_sup = load_suppliers_df()

    # 5.1 — Initialiser la clé sélection si elle n'est plus valide
    if st.session_state.sup_selected_id is not None:
        if int(st.session_state.sup_selected_id) not in set(df_sup["id"].astype(int).tolist()):
            st.session_state.sup_selected_id = None

    # 5.2 — Calcul de l'index pré-sélectionné (pour ag-Grid ou fallback)
    preselect_idx = None
    if st.session_state.sup_selected_id is not None:
        try:
            preselect_idx = int(df_sup.index[df_sup["id"].astype(int) == int(st.session_state.sup_selected_id)][0])
        except Exception:
            preselect_idx = None

    # 5.3 — Affichage tableau (ag-Grid si dispo)
    use_aggrid = False
    try:
        use_aggrid = True
    except Exception:
        use_aggrid = False

    selected_row = None

    if use_aggrid:
        gob = GridOptionsBuilder.from_dataframe(df_sup)

        gob.configure_default_column(
            filter=True,
            floatingFilter=False,          # pas de 2e rangée
            sortable=False,
            resizable=True,
            suppressMenu=False,
            menuTabs=['filterMenuTab'],    # uniquement l’onglet Filtre
            editable=False,
        )

        gob.configure_selection(selection_mode="single", use_checkbox=True)
        gob.configure_selection(selection_mode="single", use_checkbox=True)

        gob.configure_column(
            "__select__",
            headerName="☑️",
            checkboxSelection=True,
            headerCheckboxSelection=False,
            width=30,
            pinned="left",
            resizable=False,
            sortable=False,
            filter=False,
            suppressMenu=True,
            lockPosition=True,
            suppressSizeToFit=True,
        )

        # Empêcher toute case/select sur les colonnes métier, en particulier 'id'
        for col in ["id", "nom", "categorie", "adresse", "code_postal", "ville", "pays"]:
            if col in df_sup.columns:
                gob.configure_column(col, checkboxSelection=False, editable=False)

        
        gob.configure_grid_options(
            getRowId=JsCode("function(p){ return String(p.data.id); }"),
            rowSelection="single",
            deltaRowDataMode=True,
            suppressRowClickSelection=True,
            
        )

        gob.configure_pagination(enabled=True, paginationAutoPageSize=True)

        if "id" in df_sup.columns:
            gob.configure_column("id", hide=True)

        grid_options = gob.build()

        # Forcer “Filtre seul” pour toutes les colonnes hors ☑️
        for col in grid_options.get("columnDefs", []):
            if col.get("field") != "__select__":
                col["menuTabs"] = ['filterMenuTab']

        grid_resp = AgGrid(
            df_sup,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            height=350,
            key="grid_fournisseurs"
        )

        sel_rows = grid_resp.get("selected_rows", [])
        has_selection = (not sel_rows.empty) if isinstance(sel_rows, pd.DataFrame) else bool(sel_rows)

        if has_selection:
            selected_row = sel_rows.iloc[0].to_dict() if isinstance(sel_rows, pd.DataFrame) else sel_rows[0]
            st.session_state.sup_selected_id = int(selected_row["id"])
        elif preselect_idx is not None:
            selected_row = df_sup.iloc[preselect_idx].to_dict()



    # 5.4 — Ligne d’actions sous le tableau
    col_mod, col_del, col_create = st.columns([1,1,1])
    with col_mod:
        disabled = selected_row is None
        if st.button("Modifier", use_container_width=True, disabled=disabled, key="btn_edit_fourn"):
            st.session_state.sup_selected_id = int(selected_row["id"])
            st.session_state.sup_dialog_open = "edit"
            st.rerun()
    # 1) Déclenche la demande de suppression
    with col_del:
        disabled = selected_row is None
        if st.button("Supprimer", use_container_width=True, type="secondary",disabled=disabled, key="btn_del_fourn"):
            if selected_row is not None:
                st.session_state["sup_to_delete_id"] = int(selected_row["id"])
                st.session_state["sup_to_delete_name"] = selected_row["nom"]

    # 2) Rendre la confirmation PERSISTANTE (dialog ou zone inline)
    if st.session_state.get("sup_to_delete_id") is not None:
        with col_del:
            col1,col2=st.columns(2)
            with col1 :
                if st.button("✅ ", key="btn_confirm_del", use_container_width=True):
                    del_id  = int(st.session_state.get("sup_to_delete_id"))
                    del_nom = st.session_state.get("sup_to_delete_name")

                    with _get_conn_for_suppliers() as c:
                        # delete_fournisseur → audit DELETE_ROW (reversible dans Audit & Sauvegardes)
                        ok, msg = delete_fournisseur(c, del_nom, user=st.session_state.get("username", ""))

                    if ok:
                        st.toast(f"« {del_nom or del_id} » supprimé", icon="🗑️")
                        st.session_state.sup_selected_id = None
                        st.session_state["sup_to_delete_id"] = None
                        st.session_state["sup_to_delete_name"] = None
                        time.sleep(1.2)
                        st.rerun()
                    else:
                        st.toast(f"« {del_nom or del_id} » non supprimé : {msg}", icon="❌")

            with col2 :
                if st.button("❌ ", key="btn_cancel_del",use_container_width=True):
                    st.session_state["sup_to_delete_id"] = None
                    st.session_state["sup_to_delete_name"] = None

    with col_create:

        if st.button("Créer", use_container_width=True, key="btn_create_fourn"):
            st.session_state.sup_row_cache = _empty_supplier()
            st.session_state.sup_selected_id = None          
            st.session_state.sup_dialog_open = "create"
            st.rerun()

    # 5.5 — Ouvrir les dialogs si demandé
    if st.session_state.sup_dialog_open == "edit":
        row_to_edit = None
        try:
            sid = int(st.session_state.get("sup_selected_id") or -1)
            if sid in set(df_sup["id"].astype(int).tolist()):
                row_to_edit = df_sup.loc[df_sup["id"].astype(int) == sid].iloc[0].to_dict()
        except Exception:
            row_to_edit = None
        if row_to_edit is not None:
            dialog_edit_supplier(row_to_edit)

    elif st.session_state.sup_dialog_open == "create":
        dialog_create_supplier()



def resolve_pdf_to_display(map_index_to_pdfs, all_pdfs_sorted) -> Optional[str]:
    idx = st.session_state.get("selected_index_norm")
    cmd_no = st.session_state.get("selected_cmd_no")

    # 1) Sélection complète → PDF net
    if idx and cmd_no and idx in map_index_to_pdfs:
        for (n, p) in map_index_to_pdfs[idx]:
            if int(n) == int(cmd_no) and os.path.isfile(p):
                return str(Path(p).resolve())

    # 2) Sinon → s'assurer que le placeholder existe et le renvoyer (ABSOLU)
    ph = ensure_placeholder_blurred(COMMAND_DIR, all_pdfs_sorted, force=False)
    return str(Path(ph).resolve()) if ph else None



def _render_pdf_page_to_image(pdf_path: str, page_index: int, base_width: int = BASE_DISPLAY_WIDTH) -> Optional[Image.Image]:
    if not pdf_path or not os.path.isfile(pdf_path):
        return None
    try:
        doc = fitz.open(pdf_path)
        page_index = max(0, min(page_index, len(doc)-1))
        page = doc[page_index]

        page_width_pt = page.rect.width or 595
        zoom = (base_width / page_width_pt) if page_width_pt else 1.0
        mat = fitz.Matrix(zoom, zoom)

        # ➜ version compatible toutes versions
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
        except TypeError:
            # versions plus anciennes / API différente
            pix = page.get_pixmap(matrix=mat)

        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        if pix.alpha:
            img = img.convert("RGB")

        doc.close()
        return img
    except Exception as e:
        st.error(f"Erreur d’aperçu PDF: {e}")
        return None


def render_pdf_viewer(pdf_path: Optional[str], key_prefix: str = "viewer"):
    if not pdf_path:
        st.warning("Aucun PDF disponible (chemin vide).")
        return

    pdf_path = str(Path(pdf_path).resolve())
    if not os.path.isfile(pdf_path):
        st.error(f"PDF introuvable: {pdf_path}")
        return

    # 1) Unifier le flux : simple drapeau placeholder
    is_placeholder = (os.path.abspath(pdf_path) == os.path.abspath(PLACEHOLDER_PATH))

    # 2) Reset pagination si le chemin change
    last_key = f"{key_prefix}_last_path"
    if st.session_state.get(last_key) != pdf_path:
        st.session_state[last_key] = pdf_path
        st.session_state["current_page"] = 1  # on repart toujours à 1 sur changement de doc

    # 3) Ouvrir le doc et borner la pagination
    try:
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        # si placeholder, on fige la page à 1
        if is_placeholder:
            st.session_state["current_page"] = 1
        # bornes (sécurité)
        if st.session_state["current_page"] > n_pages:
            st.session_state["current_page"] = n_pages
        if st.session_state["current_page"] < 1:
            st.session_state["current_page"] = 1
        doc.close()
    except Exception as e:
        st.error(f"Impossible d’ouvrir le PDF ({pdf_path}): {e}")
        return

    # 4) Barre de navigation identique, mais désactivée si placeholder
    Buton_naviguation_1, Buton_naviguation_2, Buton_naviguation_3, Buton_naviguation_4 = st.columns([1, 1, 1, 1])
    cprev, cpage, cnext = st.columns([1, 2, 1])

    with Buton_naviguation_2:
        st.button(
            "Précédente",
            on_click=None if is_placeholder else (lambda: st.session_state.update(
                current_page=max(1, st.session_state["current_page"] - 1)
            )),
            disabled=is_placeholder or (st.session_state["current_page"] <= 1),
            key=f"{key_prefix}_prev",
            use_container_width=True,
        )

    with cpage:
        # même libellé pour tout le monde
        st.markdown(f"**Page {st.session_state['current_page']} / {n_pages}**")

    with Buton_naviguation_3:
        st.button(
            "Suivante",
            on_click=None if is_placeholder else (lambda: st.session_state.update(
                current_page=min(n_pages, st.session_state["current_page"] + 1)
            )),
            disabled=is_placeholder or (st.session_state["current_page"] >= n_pages),
            key=f"{key_prefix}_next",
            use_container_width=True,
        )

    # 5) Rendu image identique (placeholder inclus)
    img = _render_pdf_page_to_image(pdf_path, st.session_state["current_page"] - 1, base_width=BASE_DISPLAY_WIDTH)
    if img:
        st.image(img, width=BASE_DISPLAY_WIDTH)
    else:
        # fallback si le rendu image échoue
        st.info("Rendu image indisponible — affichage direct du PDF.")
        try:
            pdf_viewer(pdf_path, height=600)
        except Exception as e:
            st.error(f"Impossible d’afficher le PDF: {e}")

    # 6) Trace utile (optionnelle)
    st.caption(
        f"DEBUG → pdf: {os.path.basename(pdf_path)} | placeholder: {is_placeholder} | page: {st.session_state['current_page']}"
    )


def _index_or_none(lst, value):
    try:
        return lst.index(value) if value in lst else None
    except Exception:
        return None


def _qty_to_int(q) -> int:
    try:
        return int(round(float(q or 0)))
    except Exception:
        return int(q or 0)

def get_enfants_df_with_quantite_sql(parent_index: str) -> pd.DataFrame:
    """
    Lit bom_edges pour 'parent_index' → [{Index, Quantité_pour_parent}] et enrichit avec méta SF.
    Renvoie un DF: Index, Libellé produit, Couleur, Quantité (pour 1 parent), Unité, Commentaire.
    """
    if not parent_index:
        return pd.DataFrame(columns=["Index", "Libellé produit", "Couleur", "Quantité", "Unité", "Commentaire"])
    parent_index = _normalize_index_value(str(parent_index))

    rows = get_children(get_conn(DB_PATH), parent_index)
    if not rows:
        return pd.DataFrame(columns=["Index", "Libellé produit", "Couleur", "Quantité", "Unité", "Commentaire"])

    df_sf = load_products(Kind.SF)
    idx = {_normalize_index_value(ix): i for i, ix in enumerate(df_sf["Index"].astype(str))} if not df_sf.empty else {}

    out = []
    for r in rows:
        cidx = _normalize_index_value(r.get("Index"))
        qpp = r.get("Quantité_pour_parent")
        meta = df_sf.iloc[idx[cidx]].to_dict() if cidx in idx else {}
        out.append({
            "Index": cidx,
            "Libellé produit": meta.get("Libellé produit", meta.get("Désignation fournisseur", "")),
            "Couleur": meta.get("Couleur", ""),
            "Quantité": float(qpp or 0),
            "Unité": meta.get("Unité", meta.get("Unite", "")),
            "Commentaire": meta.get("Commentaire", "")
        })
    return pd.DataFrame(out, columns=["Index", "Libellé produit", "Couleur", "Quantité", "Unité", "Commentaire"])

def check_and_apply_stock_movements(parent_index: str, ordered_qty: float, numero_cmd: int, user: str) -> Tuple[bool, str]:
    """
    Vérifie la dispo en stock réel des enfants et applique :
      enfants : stock_real -= (qpp × ordered_qty)
      parent  : stock_virtual += ordered_qty
    """
    conn = get_conn(DB_PATH)
    parent_index = _normalize_index_value(parent_index)
    ordered_qty_i = _qty_to_int(ordered_qty)

    children = get_children(conn, parent_index)
    if not children:
        before = _init_stock_if_missing(conn, parent_index, user)
        _update_stock(conn, parent_index, new_real=None, new_virtual=before.stock_virtual + ordered_qty_i,
                      user=user, note=f"Commande achat n°{numero_cmd} (+{ordered_qty_i} virtuel)", action="edit_virtual")
        return True, "Stock virtuel parent mis à jour (pas d'enfants)."

    # 1) Check dispo
    manques, needs = [], []
    for r in children:
        child_idx = _normalize_index_value(r.get("Index"))
        q_per_parent = float(r.get("Quantité_pour_parent") or 0)
        needed = _qty_to_int(q_per_parent * float(ordered_qty))
        if needed <= 0:
            continue
        sr = _init_stock_if_missing(conn, child_idx, user)
        if sr.stock_real < needed:
            manques.append((child_idx, needed, sr.stock_real))
        needs.append((child_idx, needed))

    if manques:
        details = "\n".join([f"- {idx} : besoin {need}, dispo {have}" for idx, need, have in manques])
        return False, ("Stock insuffisant sur les composants :\n" + details)

    # 2) Transaction : débit enfants / crédit parent virtuel
    note_base = f"Commande achat n°{numero_cmd} pour {parent_index}"
    with conn:
        for child_idx, needed in needs:
            before = _init_stock_if_missing(conn, child_idx, user)
            _update_stock(conn, child_idx, new_real=before.stock_real - needed, new_virtual=None,
                          user=user, note=f"{note_base} (consommation {child_idx} : -{needed} réel)", action="edit_real")
        before_p = _init_stock_if_missing(conn, parent_index, user)
        _update_stock(conn, parent_index, new_real=None, new_virtual=before_p.stock_virtual + ordered_qty_i,
                      user=user, note=f"{note_base} (+{ordered_qty_i} virtuel parent)", action="edit_virtual")
    return True, "Mouvements de stock appliqués."

# ───────────────── Fichier : calcul du prochain LGF_XXXXX ─────────────────
# ───────────────── Fichier : calcul du prochain LGF_XXXXX ─────────────────
def compute_next_pdf_number(
    base_dir: str = COMMAND_DIR,
    prefix: str = "LGF_",
    digits: int = 5,
) -> tuple[int, str]:
    """
    Scanne base_dir pour trouver le prochain numéro disponible et
    renvoie (numero_commande, base_save_path_sans_index).
    Ex: (12, "data/commande/LGF_00012.pdf")
    """
    map_idx, all_pdfs = scan_pdf_inventory(base_dir, cache_buster=st.session_state.get("cache_buster", 0))
    if not all_pdfs:
        next_no = 1
    else:
        next_no = max(num for (num, _, _) in all_pdfs) + 1

    base_save_path = os.path.join(base_dir, f"{prefix}{next_no:0{digits}d}.pdf")
    return next_no, base_save_path



# --------------------- Utils cache & scan --------------------
@st.cache_data(show_spinner=False)
def scan_pdf_inventory(command_dir: str, cache_buster: int = 0) -> Tuple[
    Dict[str, List[Tuple[int, str]]],  # map_index_to_pdfs
    List[Tuple[int, str, str]]          # all_pdfs_sorted: (num, index_norm, path)
]:
    map_index_to_pdfs: Dict[str, List[Tuple[int, str]]] = {}
    all_pdfs: List[Tuple[int, str, str]] = []  # (num, index_norm, path)

    if not os.path.isdir(command_dir):
        return {}, []

    for name in os.listdir(command_dir):
        if not name.lower().endswith(".pdf"):
            continue
        m = PDF_PAT.match(name)
        if not m:
            continue
        num_str, raw_index = m.groups()
        try:
            num = int(num_str)
        except ValueError:
            continue
        index_norm = _normalize_index_value(raw_index)
        path = os.path.join(command_dir, name)
        map_index_to_pdfs.setdefault(index_norm, []).append((num, path))
        all_pdfs.append((num, index_norm, path))

    # Trier les listes
    for k in list(map_index_to_pdfs.keys()):
        map_index_to_pdfs[k].sort(key=lambda t: t[0], reverse=True)
    all_pdfs.sort(key=lambda t: t[0], reverse=True)

    return map_index_to_pdfs, all_pdfs



@st.cache_data(show_spinner=False)
def list_pdfs(root: Path, cache_buster: int = 0) -> list[Path]:
    return sorted([p for p in root.glob("*.pdf") if p.is_file()], key=lambda p: p.name.lower())

def scan_command_folder_map_index_to_pdfs(folder: Path):
    out = {}
    for p in folder.glob("*.pdf"):
        m = PDF_PAT.match(p.name)
        if not m:
            continue
        num = int(m.group(1))
        idx = _normalize_index_value(m.group(2))
        out.setdefault(idx, []).append((num, str(p)))
    for idx, lst in out.items():
        lst.sort(key=lambda t: t[0], reverse=True)
    return out


def list_products_with_pdfs(kind: Kind, map_index_to_pdfs: Dict[str, List[Tuple[int, str]]]):
    df = load_products(kind).copy()

    # 1) S'assurer qu'une colonne "Index" existe (aliases courants)
    if "Index" not in df.columns:
        for alt in ["INDEX", "Index produit", "Index_produit"]:
            if alt in df.columns:
                df["Index"] = df[alt]
                break
    if "Index" not in df.columns:
        # Impossible de mapper sans colonne Index
        return df.iloc[0:0], {}

    # 2) Normaliser et filtrer sur les Index qui ont au moins un PDF
    df["Index_norm"] = df["Index"].astype(str).apply(_normalize_index_value)
    df = df[df["Index_norm"].isin(map_index_to_pdfs.keys())]

    # 3) Colonnes à afficher dans la grille
    cols = ["Libellé produit", "Index", "Couleur", "Référence", "Index_norm"]
    existing = [c for c in cols if c in df.columns]
    return df[existing]

def _idx_in(lst, value, fallback=0):
    """Retourne l'index de value si présente, sinon un index de repli (0 si liste non vide), sinon None si liste vide."""
    if not lst:
        return None
    try:
        return lst.index(value) if value in lst else fallback
    except Exception:
        return fallback

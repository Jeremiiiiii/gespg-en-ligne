#Application
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import os
from datetime import date
from fpdf import FPDF
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict


from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from services.donnees import (
    load_products,
    _normalize_index_value,
    normalize_selected_list,
    filtrer_df_par_categories,
    _reload_tables_from_sql,
    Kind
)

from SQL.sql_bom import (
    get_pref, 
    set_pref,
    init_schema,                    
)

from SQL.db import DB_PATH,get_conn

from services.interaction_DB import (
    delete_rows_by_index,
    save_products

)

from services.fiche_produit import (fiche_produit_dialog,
                                    creation_fiche_produit_dialog)
                    
from services.creation_pdf_fonction import (
    MODE_PAIEMENT,
    DELAIS_PAIEMENT,
    DEFAULT_DIR,
    ROOT_DIR,
    BASE_DISPLAY_WIDTH,
    COMMAND_DIR,
    conn,
    Nom_fournisseur,
    _invalidate_pdf_cache_and_rescan,
    _on_kind_change,
    kind_key_tag,
    render_consultation_grid,
    render_selection_and_actions,
    render_pdf_viewer,
    _index_or_none,
    get_enfants_df_with_quantite_sql,
    check_and_apply_stock_movements,
    compute_next_pdf_number,
    scan_pdf_inventory,
    list_pdfs,
    resolve_pdf_to_display,
    _infos_fournisseur,
    list_products_with_pdfs,
    _idx_in,
    )

from services.audits_sauvegarde_fonctions import(

            list_actions,
            load_cell_audit,
            _edge_labels,
            list_known_users,
            delete_audit_ids,
            reset_audit_log,
            _supplier_name,
            _build_groups,
            revert_group,
            _ncols_for_changes,
            _pill_for_action
)


from services.Maj_SQL_Fonction import (
    _clean_header,
    _detect_file_type,
    _read_any_spreadsheet,
    _log_dedup,
    _count_products,
    preprocess_source_df,
    map_source_to_pf,
    assign_indexes_sequential,
    _ts_now,
    _fetch_audit_since,
    _group_changes,
    CONSOLE_KEY,

)


from services.fonction_trie_donnees import (
    ALL_FIELDS,
    NUM_FIELDS,
    DEFAULT_WEIGHTS,
    _ensure_weights_sum_to_one,
    build_graph_and_clusters_homogeneous,
    serialize_lots,
    delete_duplicates_for_lot,
    get_logged_user
)

from services.gestion_stock_fonction import (
    main
    )

from services.ui import use_base_ui

use_base_ui(page_title="Application", sidebar="collapsed",inject_scroll_lock=False)

# --- Guard d'auth : bloque l'accès si non connecté ---
if not st.session_state.get("authenticated", False):
    
    st.switch_page("Connexion.py")
    st.stop()


# ============================== SQL (BOM) ==============================

_conn = get_conn(DB_PATH)
init_schema(_conn)

#------------------------------------------------------------------------------- FONCTION ------------------------------------------------------------------------------------
# --- INITIALISATION DES DONNÉES EN MÉMOIRE (SESSION) ---
if "df_data" not in st.session_state:
    st.session_state.df_data = load_products(Kind.PF)
    
if "df_data2" not in st.session_state:
    st.session_state.df_data2 = load_products(Kind.SF)


if "modal_open" not in st.session_state:
    st.session_state.modal_open = False

# Chargement complet (df_full et df2_full) pour la logique "ArbreParenté"
df_full = st.session_state.df_data.copy()
df2_full = st.session_state.df_data2.copy()

def render_card(libelle, reference, couleur, prix, key=None):
    st.markdown(
        f"""
        <div style="padding:{PADDING_CARD}; border-radius:12px; background:#f9f9f9;
            margin-bottom:10px; box-shadow:0 2px 6px rgba(0,0,0,0.1);">
            <h4 style="margin:0; color:#2C3E50;">{libelle}</h4>
            <div style="margin:5px 0; color:#7f8c8d; display:block !important;">
                Référence : <b>{reference}</b><br>
                Couleur&nbsp;&nbsp;: <b>{couleur}</b><br>
                Prix&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <b>{prix}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

PADDING_CARD = "12px"

tab_desc, tab_gestion_produit, tab_creation_pdf,tab_maj_DB,tab_trie_donnee,tab_audit_et_sauvegarde = st.tabs(["Selection produit", "Gestion des stocks", "Création Fiche pdf","Maj DB","Trie donnée","Audit & Sauvegarde"])

#_______________________________________________________________________________ Onglet 1 __________________________________________________________________________________

width_logoGF = 250
radio_width = 153
width_container = 456

# calculer colonnes avant et après pour centrer
col_before_image = (width_container - width_logoGF) // 2
col_before_radio = (width_container - radio_width) // 2

ratio_image_left = col_before_image / width_container
ratio_image_center = width_logoGF / width_container
ratio_image_right = 1 - ratio_image_left - ratio_image_center

ratio_radio_left = col_before_radio / width_container
ratio_radio_center = radio_width / width_container
ratio_radio_right = 1 - ratio_radio_left - ratio_radio_center

with tab_desc:
  
    with st.container():

        tableau,description =st.columns([5,3])
  
        with description :

            containerdroit_photo=st.container(border=False)
            containerdroit1=st.container(border=False)
            

            with containerdroit_photo:

                st.markdown('<div class="center-wrapper">', unsafe_allow_html=True)

                cols_img = st.columns([ratio_image_left, ratio_image_center, ratio_image_right])

                with cols_img[1]:
                    st.image("components/img/headerGF.png", width=width_logoGF)

                cols_radio = st.columns([ratio_radio_left, ratio_radio_center, ratio_radio_right])
                with cols_radio[1]:
                    choix_source = st.radio("", ["Fini", "Semi"], horizontal=True, index=0)
                            
                    if choix_source == "Fini":

                        df = filtrer_df_par_categories(
                            st.session_state.df_data.copy(),
                            ['Index','Référence','Libellé produit','Composition','Couleur','Marque','Famille','Libellé famille',
                            "Prix d'achat",'PR','Unité','PV TTC','Code liaison externe','Commentaire']
                        )
                    else:  # Semi-fini

                        df = filtrer_df_par_categories(
                            st.session_state.df_data2.copy(),
                            ["Index","Libellé produit","Composition","Couleur","Unité","Fournisseur","Désignation fournisseur","PA","PR","Code liaison externe",
                             "Commentaire","Prix d'achat","Marque","Famille","Référence","PV TTC","Libellé famille"]
                            
                        )
                                # --- Gestion des sélections en session ---
                    if "ag_main_selected" not in st.session_state:
                        st.session_state["ag_main_selected"] = []  

                    st.markdown('</div>', unsafe_allow_html=True)

            with tableau :

                with st.container(border=True):
                    # --- CONFIGURATION DU TABLEAU AG-GRID ---
                    gb = GridOptionsBuilder.from_dataframe(df)

                    gb.configure_column(
                    "☑️",
                    headerCheckboxSelection=True,   
                    checkboxSelection=True,         
                    editable=False,
                    pinned="left",
                    suppressMovable=True,

                    width=40,
                    minWidth=40,
                    maxWidth=40,
                    resizable=False,               
                    suppressSizeToFit=True,        

                   
                    sortable=False,
                    filter=False,
                    suppressMenu=True,              
                    suppressHeaderMenuButton=True,  
                    suppressFloatingFilter=True    
                )


                    gb.configure_default_column(
                        editable=True,
                        filter=True,
                        sortable=True,
                        enableColumnMove=True
                    )

                    grid_options = gb.build()

                    # Restaurer les sélections si présentes
                    stored_main = normalize_selected_list(st.session_state.get("ag_main_selected", []))
                    valid_ids = set(df["Index"].astype(str).str.strip().tolist())
                    pre_selected_main = [r for r in stored_main if str(r.get("Index", "")).strip() in valid_ids]

                    grid_response = AgGrid(
                        df,
                        gridOptions=grid_options,
                        editable=True,
                        update_mode=GridUpdateMode.MODEL_CHANGED,
                        pre_selected_rows=pre_selected_main,
                        key=f"ag_main_grid_{choix_source}_{st.session_state.get('aggrid_refresh', 0)}"
,      
                        allow_unsafe_jscode=True,
                        height=500,
                        fit_columns_on_grid_load=False,
                        enable_enterprise_modules=False,

                    )

            with description:
                
                # Récupère la sélection brute depuis AgGrid
                selected_raw = grid_response.get("selected_rows", None)

                # Normalise en liste de dicts ([], si rien)
                selected = normalize_selected_list(selected_raw)

                # Persiste dans la session
                st.session_state["ag_main_selected"] = selected

                # --- Zone "toujours présente" : visuel + bouton ---
                placeholder_img = "components/img/headerGF.png"  # mets ton image ici

                with st.container():

                    if not selected:
                        render_card(" Libellé produit", " Index", " Couleur", " Prix d'achat")
                    else:
                        # Affiche un mini-résumé pour chaque ligne sélectionnée
                        for row in selected:
                                    # Identifiant de la ligne (Index ou Référence)
                                    ix = row.get("Index") or row.get("Référence")

                                    # Choisir la bonne source complète
                                    source_df = df_full if choix_source == "Fini" else df2_full
                                    keycol = "Index" if "Index" in source_df.columns else "Référence"

                                    # Récupération de la ligne complète
                                    full_row = None
                                    if ix is not None:
                                        m = source_df[source_df[keycol].astype(str).str.strip() == str(ix).strip()]
                                        if len(m):
                                            full_row = m.iloc[0].to_dict()

                                    data = full_row or row  # fallback si non trouvé

                                    # Valeurs
                                    libelle   = data.get("Libellé produit", "")
                                    reference = data.get("Index", data.get("Référence", ""))
                                    couleur   = data.get("Couleur", "")
                                    prix      = data.get("Prix d'achat", data.get("PA", ""))

                                    # Rendu
                                    render_card(libelle, reference, couleur, prix)
                        
                        # Label du bouton = info de la 1ère sélection
                        r0 = selected[0]
                        btn_label = "📄 Modifier fiche produit"
                        if st.button(btn_label, use_container_width=True, key=f"open_modal_{choix_source}"):
                            fiche_produit_dialog(selected)

                        if st.session_state.get("show_create_fp"):
                            creation_fiche_produit_dialog(choix_source)  

                    # --- Bouton supprimer sous "Création fiche produit" ---
                    if st.button("🗑️ Supprimer la sélection", use_container_width=True, key="btn_delete_rows"):
                        selected = normalize_selected_list(st.session_state.get("ag_main_selected", []))
                        indices_to_delete = [_normalize_index_value(r.get("Index","")) for r in selected if r.get("Index")]
                        if not indices_to_delete:
                            st.warning("Coche au moins une ligne à supprimer.")
                        else:
                            if choix_source == "Fini":
                                df_key = "df_data"; kind = Kind.PF
                            else:
                                df_key = "df_data2"; kind = Kind.SF
                            delete_rows_by_index(df_key, kind, indices_to_delete)


#___________________________________________________________________________Gestion Stock __________________________________________________________________________________


with tab_gestion_produit:
    main()

#___________________________________________________________________________Création PDF __________________________________________________________________________________
 
with tab_trie_donnee:

    # ============================== ETAT (SESSION) =================================
    st.session_state.setdefault("sim_results", None)
    st.session_state.setdefault("selected_lots", set())
    st.session_state.setdefault("hide_resolved", True)
    st.session_state.setdefault("pending_confirm", None)  # id du lot en attente de confirmation (bouton ligne)
    st.session_state.setdefault("max_lots", 200) 

    # ============================== UI — EN-TÊTE ==================================
    st.title("Tri par similarité (PF↔PF & SF↔SF)")
    st.caption("100% = toutes les catégories identiques (hors index). Pondération modifiable. Actions par lot & en masse. Pas d'export.")

    with get_conn() as conn:
        df_all = pd.read_sql_query("SELECT rowid, * FROM products;", conn)
    for col in NUM_FIELDS:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    df_pf = df_all[df_all["kind"] == "PF"].reset_index(drop=True).copy()
    df_sf = df_all[df_all["kind"] == "SF"].reset_index(drop=True).copy()

    c1, c2, c3 = st.columns([2, 3, 7])
    with c1:
        with st.container(border=True):
            seuil = st.slider("Seuil (%)", 70, 100, 90, 1)
            threshold = seuil / 100.0
    with c2:
        with st.container(border=True):
            st.markdown("")
            adv = st.checkbox("Pondérations avancées", value=False)
            st.markdown("")
            st.markdown("")
    with c3:
        with st.container(border=True):
            max_lots = st.number_input(
                "Plafond de lots (max)",
                min_value=10, max_value=10_000,
                value=int(st.session_state.get("max_lots", 200)),
                step=10,
                help="Nombre maximum de lots à afficher pour éviter les ralentissements."
            )
            st.session_state["max_lots"] = int(max_lots)

    if adv:
        with st.expander("⚙️ Pondérations (avancées)", expanded=True):
            w = {}
            cols = st.columns(4)
            for i, f in enumerate(ALL_FIELDS):
                with cols[i % 4]:
                    w[f] = st.number_input(f"Poids — {f}", 0.0, 1.0, float(DEFAULT_WEIGHTS.get(f, 0.0)), 0.01, key=f"w_{f}")
        weights = _ensure_weights_sum_to_one(w)
    else:
        weights = DEFAULT_WEIGHTS

    # Bouton d'analyse
    analyze = st.button("🔎 Analyser les similarités", type="primary")

    if analyze:
        with st.spinner("Construction des lots…"):
            e_pf, lots_pf = build_graph_and_clusters_homogeneous(df_pf, weights, threshold, "PF")
            e_sf, lots_sf = build_graph_and_clusters_homogeneous(df_sf, weights, threshold, "SF")

            raw_lots = lots_pf + lots_sf
            total_found = len(raw_lots)
            max_lots = int(st.session_state.get("max_lots", 200))
            trimmed = max(total_found - max_lots, 0)
            if trimmed > 0:
                raw_lots = raw_lots[:max_lots]  # 🚦 application du plafond

            lots_serialized = serialize_lots(raw_lots, df_pf, df_sf)

        st.session_state["sim_results"] = {
            "threshold": threshold,
            "weights": weights,
            "lots": lots_serialized,
            "total_found": total_found,
            "max_lots": max_lots,
            "trimmed": trimmed,
        }
        st.session_state["selected_lots"] = set()
        st.session_state["pending_confirm"] = None

    # ============================== VUE RÉSULTATS =================================

    res = st.session_state.get("sim_results")

    if not res:
        st.info("Réglez le seuil / poids, puis cliquez sur **Analyser les similarités**.")
    else:
        if res.get("trimmed", 0) > 0:
            st.warning(
                f"Beaucoup de lots détectés ({res['total_found']}). "
                f"Affichage limité à {res['max_lots']} pour de meilleures performances. "
                f"Augmentez le plafond si nécessaire."
            )

        def signature_label(sig: Tuple[str, ...]) -> str:
            return ", ".join(sig) if sig else "— (aucune différence) —"

        def purge_resolved_from_results():
            """Retire définitivement les lots résolus de l'écran (pas juste masqués)."""
            r = st.session_state.get("sim_results")
            if not r: return
            before = len(r["lots"])
            r["lots"] = [l for l in r["lots"] if not l.get("resolved")]
            after = len(r["lots"])
            st.session_state["sim_results"] = r
            return before - after

        # Filtre “masquer résolus” (conservé pour info) mais on purgera quand même après suppression
        hide_resolved = st.toggle("Masquer les lots déjà traités", value=st.session_state.get("hide_resolved", True))
        st.session_state["hide_resolved"] = hide_resolved

        lots_all: List[Dict[str, Any]] = res["lots"]
        lots_to_show = [l for l in lots_all if not (hide_resolved and l.get("resolved"))]

        # Stats
        total_pairs = sum(max(len(l["members"]) - 1, 0) for l in lots_to_show)
        cA, cB, cC, cD = st.columns(4)
        with cA: st.metric("Lots visibles", len(lots_to_show))
        with cB: 
            avg_size = (sum(len(l['members']) for l in lots_to_show)/len(lots_to_show)) if lots_to_show else 0
            st.metric("Taille moyenne", f"{avg_size:.2f}")
        with cC: st.metric("Paires implicites (≈)", total_pairs)
        with cD:
            if st.button("♻️ Réinitialiser l'analyse"):
                st.session_state["sim_results"] = None
                st.session_state["selected_lots"] = set()
                st.session_state["pending_confirm"] = None
                st.rerun()

        st.markdown("---")

        # ============================== BARRE D'ACTIONS DE MASSE ======================
        user = get_logged_user()
        left, mid, right = st.columns([6,3,3])
        with left:
            selected_ids: Set[int] = set(st.session_state.get("selected_lots", set()))
            st.markdown(f"**Sélection :** {len(selected_ids)} lot(s)")

        with mid:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Tout sélectionner"):
                    st.session_state["selected_lots"] = {l["id"] for l in lots_to_show if not l.get("resolved")}
                    st.rerun()
            with c2:
                if st.button("Tout désélectionner"):
                    st.session_state["selected_lots"] = set()
                    st.rerun()

        with right:
            confirm_mass = st.checkbox("Confirmer la suppression", value=False, key="confirm_mass")
            if st.button("✂️ Supprimer les doublons (sélection)", type="primary", disabled=not confirm_mass or len(st.session_state["selected_lots"]) == 0):
                # On itère pour produire des toasts par statut et on purge du state
                successes = partials = failures = 0
                total_deleted = 0
                acted = 0
                for lot in list(res["lots"]):  # copie défensive
                    if lot["id"] in st.session_state["selected_lots"] and not lot.get("resolved"):
                        acted += 1
                        expected = max(len(lot["members"]) - 1, 0)
                        n = delete_duplicates_for_lot(lot, user=user)
                        total_deleted += n
                        if expected == 0 or n == expected:
                            successes += 1
                        elif n == 0 and expected > 0:
                            failures += 1
                        else:
                            partials += 1
                purged = purge_resolved_from_results()
                st.session_state["selected_lots"] = set()
                # Toast synthèse
                if acted == 0:
                    st.toast("Aucune action : les lots sélectionnés étaient déjà traités.", icon="⚠️")
                else:
                    msg = f"Suppression terminée : {successes} OK, {partials} partiels, {failures} échecs • {total_deleted} enregistrement(s) supprimé(s)."
                    icon = "✅" if failures == 0 and partials == 0 else ("⚠️" if partials > 0 else "❌")
                    st.toast(msg, icon=icon)
                st.rerun()

        # ============================== GROUPEMENT PAR DIFFÉRENCES ====================
        st.markdown("### Groupes de lots par différences communes")
        st.caption("Chaque groupe correspond à une *signature* (les mêmes colonnes qui diffèrent). Dépliez pour voir les lots et détails.")

        # Constitution des groupes
        groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        for l in lots_to_show:
            sig = tuple(l.get("signature") or tuple())
            groups[sig].append(l)

        # Tri : groupes les plus volumineux d'abord
        def group_stats(lots_group: List[Dict[str, Any]]) -> Tuple[int, int]:
            return len(lots_group), sum(len(x["members"]) for x in lots_group)

        ordered_groups = sorted(groups.items(), key=lambda kv: (len(kv[1]), group_stats(kv[1])[1]), reverse=True)

        # Rendu
        for g_idx, (sig_tuple, group_lots) in enumerate(ordered_groups):
            glen, gmembers = group_stats(group_lots)
            sig_txt = signature_label(sig_tuple)

            with st.container():
                # En-tête de groupe (carte)
                hdr_left, hdr_mid, hdr_right = st.columns([6,3,3])
                with hdr_left:
                    st.markdown(f"**Différences :** {sig_txt}")
                    st.markdown(
                        f"<span class='badge warn'>{glen} lot(s)</span> "
                        f"<span class='badge'>{gmembers} élément(s) au total</span>",
                        unsafe_allow_html=True
                    )
                with hdr_mid:
                    if st.button("Sélectionner le groupe", key=f"selgrp_{g_idx}"):
                        ids = {lot["id"] for lot in group_lots if not lot.get("resolved")}
                        st.session_state["selected_lots"].update(ids)
                        st.rerun()
                with hdr_right:
                    if st.button("Désélectionner le groupe", key=f"unselgrp_{g_idx}"):
                        ids = {lot["id"] for lot in group_lots}
                        st.session_state["selected_lots"].difference_update(ids)
                        st.rerun()

                # Dépliant : détails des lots du groupe
                with st.expander(f"Afficher les {glen} lot(s)", expanded=False):
                    for lot in group_lots:
                        lid = lot["id"]
                        tag = lot["tag"]
                        size = len(lot["members"])
                        resolved = lot.get("resolved", False)

                        col1, col2, col3 = st.columns([1.0, 6.0, 3.0])

                        # Sélection lot
                        with col1:
                            sel = (lid in st.session_state["selected_lots"])
                            if st.checkbox("", key=f"sel_lot_{lid}", value=sel):
                                st.session_state["selected_lots"].add(lid)
                            else:
                                st.session_state["selected_lots"].discard(lid)

                        # Infos lot
                        with col2:
                            bclass = "pf" if tag == "PF" else "sf"
                            st.markdown(
                                f"<span class='badge {bclass}'>{tag}</span> "
                                f"<span class='badge warn'>{size} élément(s)</span> "
                                f"<span class='small'>→ Conserver: <code>{lot['keep_index']}</code></span>",
                                unsafe_allow_html=True
                            )
                            if resolved:
                                st.markdown("<span class='badge ok'>résolu</span>", unsafe_allow_html=True)

                            # Détails membres (sous-dépliant)
                            with st.expander("Voir les membres", expanded=False):
                                df_preview = pd.DataFrame(lot["members"])[["product_index","reference","libelle_produit","marque","couleur","unite"]]
                                st.dataframe(df_preview, use_container_width=True, hide_index=True, height=min(40 + 24*len(df_preview), 160))

                        # Action (suppression du lot)
                        with col3:
                            pending = st.session_state.get("pending_confirm")
                            if pending == lid:
                                cL, cR = st.columns(2)
                                with cL:
                                    if st.button("✅ Confirmer", key=f"confirm_{lid}", use_container_width=True):
                                        expected = max(len(lot["members"]) - 1, 0)
                                        n = delete_duplicates_for_lot(lot, user=user)
                                        # Purge immédiate des notifs supprimées
                                        purged = purge_resolved_from_results()
                                        # Toast de résultat
                                        if expected == 0 or n == expected:
                                            st.toast(f"Lot {lid} : doublons supprimés ({n}/{expected}).", icon="✅")
                                        elif n == 0 and expected > 0:
                                            st.toast(f"Lot {lid} : échec de suppression ({n}/{expected}).", icon="❌")
                                        else:
                                            st.toast(f"Lot {lid} : suppression partielle ({n}/{expected}).", icon="⚠️")
                                        st.session_state["pending_confirm"] = None
                                        st.rerun()
                                with cR:
                                    if st.button("↩️ Annuler", key=f"cancel_{lid}", use_container_width=True):
                                        st.session_state["pending_confirm"] = None
                                        st.rerun()
                            else:
                                if resolved:
                                    st.write("—")
                                else:
                                    if st.button("✂️ Éliminer ce lot", key=f"del_{lid}", use_container_width=True):
                                        st.session_state["pending_confirm"] = lid
                                        st.rerun()

                    st.markdown("<hr class='soft'/>", unsafe_allow_html=True)




with tab_creation_pdf:
    
    # ==== État persistant pour la grille fournisseurs ====
    if "sup_selected_id" not in st.session_state:
        st.session_state.sup_selected_id = None

    if "sup_dialog_open" not in st.session_state:
        st.session_state.sup_dialog_open = None  # "edit" | "create" | None

    if "sup_row_cache" not in st.session_state:
        st.session_state.sup_row_cache = {}

    if "cache_buster" not in st.session_state:
        st.session_state["cache_buster"] = 0
    if "current_kind" not in st.session_state:
        st.session_state["current_kind"] = Kind.PF
    if "selected_index_norm" not in st.session_state:
        st.session_state["selected_index_norm"] = None
    if "selected_index_raw" not in st.session_state:
        st.session_state["selected_index_raw"] = None
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = None

        

    # ---------------------- Liste des PDF ----------------
    pdf_paths = list_pdfs(ROOT_DIR, cache_buster=st.session_state["cache_buster"])
    st.caption(f"{len(pdf_paths)} PDF trouvé(s) dans `{DEFAULT_DIR}`")

    # -------------------- Layout principal ----------------
    left, right = st.columns([2, 3], gap="large")
    with left :
        col_left_top_1,col_left_top_2,col_left_top_3=st.columns([1,1,1])
        with col_left_top_2:
                    
                chosen = st.radio(
                "Type pour la liste :",
                options=[Kind.PF, Kind.SF],
                format_func=lambda k: "PF" if k == Kind.PF else "SF",
                horizontal=True,
                key="grid_kind_obj",
            )
                st.markdown("")

    # ====== Colonne droite (top) : radio PF/SF ========

    with right:
        col_right_top_1, col_right_top_2, col_right_top_3 = st.columns([5, 4, 3])
        with col_right_top_2:

            chosen_enum = chosen 
        

            if not isinstance(st.session_state.get("current_kind"), Kind):
                st.session_state["current_kind"] = Kind.PF  

            if chosen_enum != st.session_state.get("current_kind", Kind.PF):
                _on_kind_change(chosen_enum)
            else:
                st.session_state["current_kind"] = chosen_enum


    map_index_to_pdfs, all_pdfs_sorted = scan_pdf_inventory(COMMAND_DIR, cache_buster=st.session_state["cache_buster"])

    # ====== Colonne gauche : Sélection + actions (PF & SF) ========
    with left:
        kind_obj = st.session_state.get("current_kind", Kind.PF)
        kind_tag = kind_key_tag(kind_obj)

        # 1) Construire la grille à partir des produits qui ONT au moins un PDF
        df_products = list_products_with_pdfs(kind_obj, map_index_to_pdfs)

        # 2) Rendu agGrid identique PF/SF
        st.caption(f"{len(df_products)} produit(s) avec PDF pour { 'PF' if kind_obj == Kind.PF else 'SF' }")
        render_consultation_grid(df_products, key_prefix=f"{kind_tag}_grid")

        # 3) Selectbox des numéros + Télécharger/Supprimer (avec confirmation)
        render_selection_and_actions(map_index_to_pdfs, key_prefix=f"{kind_tag}_sel")

    # ====== Colonne droite : Aperçu PDF (PF & SF) avec flou auto ========
    with right:

        pdf_path = resolve_pdf_to_display(map_index_to_pdfs, all_pdfs_sorted)
        kind_obj = st.session_state.get("current_kind", Kind.PF)
        kind_tag = kind_key_tag(kind_obj)
        render_pdf_viewer(pdf_path, key_prefix=f"{kind_tag}_viewer")
    # ───────────────── UI : formulaire ─────────────────
    expander = st.expander(label="formulaire création commande")
    st.markdown("")
    st.markdown("")
    conn = get_conn()
    last_cmd = get_pref(conn, "last_id_commandeur")
    last_recv = get_pref(conn, "last_id_receveur")
    last_fourn = get_pref(conn, "last_id_fournisseur")
    idx_commandeur_def = _index_or_none(Nom_fournisseur, last_cmd)
    idx_receveur_def   = _index_or_none(Nom_fournisseur, last_recv)
    idx_fourn_def      = _index_or_none(Nom_fournisseur, last_fourn)

    with expander:
        with st.container(border=True):
            st.write("Création pdf")
            id_commandeur = st.selectbox(
                                "Identité de la personne qui commande",
                                Nom_fournisseur,
                                index=_idx_in(Nom_fournisseur, last_cmd, fallback=0),
                                key="id_commandeur"
                            )
            if id_commandeur is not None:
                infos = _infos_fournisseur(id_commandeur)
                ville_commandeur = infos["ville"]
                adresse_commandeur = infos["adresse"]
                code_postal_commandeur = infos["code_postal"]

            id_receveur = st.selectbox(
                                "À livrer à l'adresse suivante",
                                Nom_fournisseur,
                                index=_idx_in(Nom_fournisseur, last_recv, fallback=0),
                                key="id_receveur"
                            )

            if id_receveur is not None:
                infos = _infos_fournisseur(id_receveur)
                ville_receveur = infos["ville"]
                adresse_receveur = infos["adresse"]
                code_postal_receveur = infos["code_postal"]
                pays_receveur = infos["pays"]
            id_fournisseur = st.selectbox(
                                    "Informations du fournisseur",
                                    Nom_fournisseur,
                                    index=_idx_in(Nom_fournisseur, last_fourn, fallback=0),
                                    key="id_fournisseur"
                                )
            if id_fournisseur is not None:
                infos = _infos_fournisseur(id_fournisseur)
                ville_fournisseur = infos["ville"]
                adresse_fournisseur = infos["adresse"]
                code_postal_fournisseur = infos["code_postal"]
                pays_fournisseur = infos["pays"]
            current_user = st.session_state.get("current_user")

            try:
                if id_commandeur and id_commandeur != last_cmd:
                    set_pref(conn, "last_id_commandeur", id_commandeur, user=current_user)
                if id_receveur and id_receveur != last_recv:
                    set_pref(conn, "last_id_receveur", id_receveur, user=current_user)
                if id_fournisseur and id_fournisseur != last_fourn:
                    set_pref(conn, "last_id_fournisseur", id_fournisseur, user=current_user)
            except Exception as e:
                pass
                
            col2, col3, col4 = st.columns(3)
            with col2:
                st.markdown('<div id="commande-form">', unsafe_allow_html=True)
                date_saisie = st.date_input("Date de la commande", value=date.today(), format="DD/MM/YYYY")
                date_commande = date_saisie.strftime("%d/%m/%Y")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                Selection_mode_de_paiement = st.selectbox("Mode de paiement", MODE_PAIEMENT, index=None)
            with col4:
                Selection_delai_de_paiement = st.selectbox("Délai de paiement (en Jrs)", DELAIS_PAIEMENT, index=None)

            Selection_saison = st.text_input("Entrez la saison", key="Selection_saison")

            col7, col8 = st.columns(2)
            with col7:
                type_produit = st.radio("Type de produit :", ["Fini", "Semi-fini"])

            # Produits depuis SQL
            liste_produit = load_products(Kind.PF if type_produit == "Fini" else Kind.SF)

            with col8:
                choix_mode = st.radio("Sélectionner par :", ["Index", "Libellé"])
            if choix_mode == "Index":
                index_selectionne = st.selectbox("Choisir un produit (Index) :", liste_produit["Index"].tolist())
            else:
                libelle_selectionne = st.selectbox("Choisir un produit (Libellé) :", liste_produit["Libellé produit"].tolist())
                index_selectionne = liste_produit.loc[liste_produit["Libellé produit"] == libelle_selectionne, "Index"].values[0]

        Index_produit = index_selectionne
        produit_selectionne = liste_produit.loc[liste_produit["Index"] == Index_produit].iloc[0]

        Libelle_produit = str(produit_selectionne.get("Libellé produit", ""))
        Composition_produit = str(produit_selectionne.get("Composition", ""))
        Couleur_produit = str(produit_selectionne.get("Couleur", ""))
        Unité_produit = str(produit_selectionne.get("Unité", ""))

        # PR comme prix unitaire
        try:
            Selection_prix_unitaire = float(produit_selectionne.get("PR", 0) or 0)
        except Exception:
            Selection_prix_unitaire = 0.0

        Selection_quantité = st.number_input("Quantité commandée", step=0.01, value=0.0, min_value=0.0, key="Selection_quantité")
        Commentaire = st.text_area("Saisissez votre commentaire ici (détaille des tailles etc)", height=150, placeholder="Tapez votre texte ici...")


    # ───────────────── PDF ─────────────────
    class CommandePDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_x(-15)
            self.set_font('DejaVu', '', 8)
            self.cell(0, 10, f' {self.page_no()}/{{nb}}', 0, 0, 'C')

        def header(self):
            self.set_font("DejaVu", "B", 11)
            self.cell(0, 10, id_commandeur or "", ln=True)
            self.set_font("DejaVu", "", 9)
            self.cell(0, 6, adresse_commandeur or "", ln=True)
            self.cell(0, 6, f"{code_postal_commandeur} {(ville_commandeur or '').upper()}", ln=True)
            img_width = 30
            try:
                self.image("components/img/headerGF.png", x=210 - self.r_margin - img_width, y=self.get_y()-15, w=img_width)
            except Exception:
                pass
            self.ln(7)
            self.line(self.get_x(), self.get_y(), 210 - self.r_margin, self.get_y())
            self.ln(7)

        def add_commande_header(self, numero_commande: int):
            self.set_font("DejaVu", "B", 11)
            self.cell(0, 8, f"COMMANDE ACHAT N° {numero_commande}", ln=True)
            self.set_font("DejaVu", "", 9)
            self.cell(0, 6, f"Date de commande : {date_commande}", ln=True)
            self.ln(7)

        def add_adresses(self):
            self.line(self.get_x(), self.get_y(), 210 - self.r_margin, self.get_y())
            self.ln(4)
            col_width = 95; col_height = 6
            self.set_font("DejaVu", "B", 11)
            self.cell(col_width, col_height, "A LIVRER A L'ADRESSE SUIVANTE", 0, 0, 'L')
            self.cell(col_width, col_height, "FOURNISSEUR", 0, 1, 'L')
            self.set_font("DejaVu", "", 9)
            self.cell(col_width, col_height, id_receveur or "", 0, 0, 'L')
            self.cell(col_width, col_height, id_fournisseur or "", 0, 1, 'L')
            self.cell(col_width, col_height, adresse_receveur or "", 0, 0, 'L')
            self.cell(col_width, col_height, adresse_fournisseur or "", 0, 1, 'L')
            self.cell(col_width, col_height, f"{code_postal_receveur} {(ville_receveur or '').upper()}", 0, 0, 'L')
            self.cell(col_width, col_height, f"{code_postal_fournisseur} {(ville_fournisseur or '').upper()}", 0, 1, 'L')
            self.cell(col_width, col_height, pays_receveur or "", 0, 0, 'L')
            self.cell(col_width, col_height, pays_fournisseur or "", 0, 1, 'L')
            self.ln(4)
            self.line(self.get_x(), self.get_y(), 210 - self.r_margin, self.get_y())
            self.ln(4)

        def add_details_table(self):
            self.set_font("DejaVu", "B", 11)
            self.multi_cell(0, 8, "INFORMATIONS DE PAIEMENT :", align="L")
            self.set_font("DejaVu", "", 9)
            self.multi_cell(0, 6, f"Mode de paiement : {Selection_mode_de_paiement}")
            self.multi_cell(0, 6, f"Délai de paiement : {Selection_delai_de_paiement}")
            self.multi_cell(0, 6, f"Saison : {Selection_saison}")
            self.ln(8)

            self.set_fill_color(240, 240, 240)
            self.set_font("DejaVu", "B", 11)
            self.multi_cell(0, 8, "DÉTAILS DU PRODUIT :", align="L", fill=True)

            self.set_font("DejaVu", "", 9)
            self.multi_cell(0, 6, f"Index : {Index_produit}", fill=True)
            self.multi_cell(0, 6, f"Libellé produit : {Libelle_produit}", fill=True)
            self.multi_cell(0, 6, f"Composition : {Composition_produit}", fill=True)
            self.multi_cell(0, 6, f"Couleur : {Couleur_produit}", fill=True)
            self.multi_cell(0, 6, f"Prix unitaire : {Selection_prix_unitaire:.2f} €", fill=True)
            self.multi_cell(0, 6, f"Date de livraison : {date_saisie}", fill=True)
            self.ln(8)

            montant_total = float(Selection_quantité) * float(Selection_prix_unitaire)
            self.set_font("DejaVu", "B", 11)
            self.cell(95, 10, "Quantité commandée", 1, 0, 'C', fill=True)
            self.cell(95, 10, "Montant total HT", 1, 1, 'C', fill=True)
            self.set_font("DejaVu", "", 9)
            self.cell(95, 10, f"{Selection_quantité} {Unité_produit}", 1, 0, 'C')
            self.cell(95, 10, f"{montant_total:.2f} €", 1, 1, 'C')
            self.ln(8)

        def _draw_stock_table_header(self, col_widths):
            self.set_fill_color(220, 220, 220)
            self.set_font("DejaVu", "B", 11)
            headers = ["Index","Libellé produit","Couleur","Quantité","Unité","Commentaire"]
            for i, h in enumerate(headers):
                self.cell(col_widths[i], 8, h, 1, 0, 'C', fill=True)
            self.ln()
            self.set_font("DejaVu", "", 9)

        def add_stock_section(self, parent_index):
            enfants_df = get_enfants_df_with_quantite_sql(parent_index)
            if enfants_df is None or enfants_df.empty:
                return

            # Règle demandée : si ≥ 3 enfants, on commence la section sur une nouvelle page
            if len(enfants_df) >= 3:
                self.add_page()

            self.set_font("DejaVu", "B", 11)
            self.set_x(self.l_margin)
            self.cell(0, 6, "STOCK A METTRE EN OEUVRE", ln=True, align="C")
            self.ln(7)

            col_widths = [23, 55, 25, 25, 18, 44]
            self._draw_stock_table_header(col_widths)

            # Impression ligne par ligne avec contrôle de débordement
            line_h_base = 6
            for _, row in enfants_df.iterrows():
                data = [
                    str(row["Index"]),
                    str(row["Libellé produit"]),
                    str(row["Couleur"]),
                    f"{float(row['Quantité']) * float(Selection_quantité)}",
                    str(row["Unité"]),
                    str(row["Commentaire"]),
                ]
                # 1) calcul des wraps sans écrire
                wrapped_texts, heights = [], []
                for i, text in enumerate(data):
                    w = col_widths[i] - 2
                    # on utilise multi_cell(..., split_only=True) pour mesurer
                    lines = self.multi_cell(w, line_h_base, text, split_only=True)
                    wrapped_texts.append(lines)
                    heights.append(len(lines) * line_h_base)
                row_h = max(heights) if heights else line_h_base

                # 2) si ça déborde, nouvelle page + réafficher en-tête
                if self.get_y() + row_h > self.page_break_trigger:
                    self.add_page()
                    self.set_font("DejaVu", "B", 11)
                    self.set_x(self.l_margin)
                    self.cell(0, 6, "STOCK A METTRE EN OEUVRE", ln=True, align="C")
                    self.ln(7)
                    self._draw_stock_table_header(col_widths)

                # 3) écrire la ligne
                x_start, y_start = self.get_x(), self.get_y()
                for i, lines in enumerate(wrapped_texts):
                    w = col_widths[i]
                    x, y = self.get_x(), self.get_y()
                    cell_h = len(lines) * line_h_base
                    offset_y = (row_h - cell_h) / 2
                    self.set_xy(x, y + offset_y)
                    for ln in lines:
                        self.cell(w, line_h_base, ln, ln=2, align='C')
                    # cadre + avancer
                    self.set_xy(x + w, y)
                    self.rect(x, y, w, row_h)
                self.set_xy(x_start, y_start + row_h)

        def add_commentaires(self):
            self.ln(7)
            self.set_font("DejaVu", "B", 11)
            self.cell(0, 6, "COMMENTAIRES", ln=True)
            self.set_font("DejaVu", "", 9)
            self.cell(0, 6, str(Commentaire), ln=True)


    # ───────────────── App : génération PDF ─────────────────

    with expander:

        colgenerate1, colgenerate2, colgenerate3 = st.columns(3)

        with colgenerate2:
            with st.form("formulaire_commande"):
                st.write("Appuyez sur le bouton pour générer le PDF.")
                generate = st.form_submit_button("Générer le PDF", width=BASE_DISPLAY_WIDTH)

            if generate:
                # Contrôles
                if not id_commandeur:               st.warning("⚠️ Vous devez sélectionner **l'identité du commandeur**")
                if not id_receveur:                 st.warning("⚠️ Vous devez sélectionner **l'adresse de livraison**")
                if not id_fournisseur:              st.warning("⚠️ Vous devez sélectionner **le fournisseur**")
                if not Selection_mode_de_paiement:  st.warning("⚠️ Vous devez choisir un **mode de paiement**")
                if not Selection_delai_de_paiement: st.warning("⚠️ Vous devez choisir un **délai de paiement**")
                if not Selection_saison:            st.warning("⚠️ Vous devez entrer une **saison**")
                if not Index_produit:               st.warning("⚠️ Vous devez choisir un **produit**")
                if Selection_quantité <= 0:         st.warning("⚠️ La **quantité** doit être > 0")

                # 1) Déterminer le prochain n° + chemin (base sans suffixe d'index)
                numero_commande, base_save_path = compute_next_pdf_number(
                    base_dir="data/commande", prefix="LGF_", digits=5
                )

                current_user = st.session_state.get("current_user")

                # 1bis) Ajouter l'Index produit à la fin du nom de fichier
                idx_norm = _normalize_index_value(str(Index_produit))

                # base_save_path = data/commande/LGF_00012.pdf  -> on remplace l'extension
                save_dir = os.path.dirname(base_save_path)
                base_name = os.path.splitext(os.path.basename(base_save_path))[0]  # LGF_00012
                save_path = os.path.join(save_dir, f"{base_name}__{idx_norm}.pdf")

                # 2) Vérifier & appliquer les mouvements de stock (avant la création du PDF)
                ok, msg = check_and_apply_stock_movements(Index_produit, Selection_quantité, numero_commande, user=current_user)
                if not ok:
                    st.error(msg)
                    st.stop()
                else:
                    st.toast(f"N° commande : {numero_commande:05d} — {msg}", icon="✅")

                # 3) Générer le PDF
                pdf = CommandePDF()
                pdf.add_font("DejaVu", "", "components/img/DejaVuSans.ttf", uni=True)
                pdf.add_font("DejaVu", "B", "components/img/DejaVuSans-Bold.ttf", uni=True)
                pdf.set_font("DejaVu", "", 11)
                pdf.alias_nb_pages()
                pdf.add_page()
                pdf.add_commande_header(numero_commande)
                pdf.add_adresses()
                pdf.add_details_table()
                pdf.add_stock_section(Index_produit)
                pdf.add_commentaires()

                # 4) Sauvegarder sous data/commande/LGF_XXXXX.pdf + bouton de téléchargement
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                with open(save_path, "wb") as f:
                    f.write(pdf_bytes)

                _invalidate_pdf_cache_and_rescan(select_idx_norm=idx_norm, select_cmd_no=numero_commande)

    
 
with tab_maj_DB: 

    #───────────────────────────────────────────────────────UI─────────────────────────────────────────────────────────────
    containerimportsheet =st.container(border=True)

    # === Import fichier ===
    with containerimportsheet:
        st.caption("Entrée identique à l'ancienne page : CSV/TXT/XLS/XLSX/XLSM")
        uploaded = st.file_uploader("Fichier d'entrée", type=["csv","txt","xls","xlsx","xlsm"], accept_multiple_files=False)

    # Sélection de feuille (si Excel)
    sheet_selected: Optional[str] = None


    if uploaded is not None and _detect_file_type(uploaded) in ("xls","xlsx","xlsm"):
        try:
            containerfeuilleaimporter =st.container(border=True)
            with containerfeuilleaimporter:
                default_brand = st.text_input("Marque par défaut (optionnel)", value="")
                uploaded.seek(0)
                xls = pd.ExcelFile(uploaded, engine="openpyxl")
                sheet_selected = st.selectbox("Feuille à importer", xls.sheet_names, index=0)
        except Exception:
            st.warning("Impossible de lister les feuilles – on utilisera la première.")
            sheet_selected = None

    if uploaded is not None:
        try:
            df_raw = _read_any_spreadsheet(uploaded, sheet=sheet_selected)

        except Exception as e:
            st.error(f"Lecture impossible : {type(e).__name__}: {e}")
            df_raw = None  # et on n’exécute pas la suite

        if uploaded is None or df_raw is None:
            st.stop()

        # Prétraitement
        df_clean, stats = preprocess_source_df(df_raw)
        with containerfeuilleaimporter:
            st.success(f"Pré-traitement : {stats['non_empty_rows']} lignes non vides → {stats['after_rows']} après dédoublonnage (clé: {stats['dedup_on']})")
        c1, c2, c3, c4,c5 = st.columns(5)
        with c2:
            with st.container(border=True):
                st.metric("Lignes non vides", stats['non_empty_rows'])

        with c3:
            with st.container(border=True):
                st.metric("Doublons supprimés", stats['dropped_duplicates'])

        with c4:
            with st.container(border=True):
                st.metric("Lignes importées", stats['after_rows'])
        
        name = getattr(uploaded, "name", "?")
        _log_dedup(
            "import",
            (name, len(df_raw)),  # signature = (fichier, nb_lignes_brutes)
            f"Fichier importé: {name} ({len(df_raw)} lignes)"
        )

        _log_dedup(
            "clean",
            (name, sheet_selected, stats['non_empty_rows'], stats['dropped_duplicates'], stats['after_rows']),
            f"Source nettoyée: non vides={stats['non_empty_rows']}, doublons supprimés={stats['dropped_duplicates']}, lignes gardées={stats['after_rows']} (clé: {stats['dedup_on']})"
        )

        base_count = _count_products()
        _log_dedup(
            "base",
            (base_count,),  # signature = (volume base)
            f"Base chargée ({base_count} lignes)"
        )


        # ===== Commentaires (même design : blocs multiples, 4 par ligne) =====
        with st.container(border=True):
            comment_mode = st.radio("Commentaires", options=["Aucun","Global","Spécifique (multi par libellé)"], index=0, horizontal=True)
            global_comment = ""
            comment_blocks: List[Dict[str, Any]] = []
        if comment_mode == "Global":
            with st.container(border=True):
                global_comment = st.text_area("Commentaire global", value="")
        elif comment_mode == "Spécifique (multi par libellé)":
            with st.container(border=True):
                # options de libellés depuis la source nettoyée
                norm = {_clean_header(c): c for c in df_clean.columns}
                lib_col = norm.get("libelleproduit")
                lib_values = sorted([s for s in (df_clean[lib_col].astype(str).unique().tolist() if lib_col else []) if s.strip() != ""])
                n_blocks = st.number_input("Nombre de blocs (commentaires différents)", min_value=1, max_value=100, value=1, step=1)
                for start in range(0, int(n_blocks), 4):
                    cols = st.columns(min(4, int(n_blocks) - start))
                    for i, col in enumerate(cols):
                        idx = start + i
                        with col:
                            sel = st.multiselect(f"Libellé(s) bloc {idx+1}", options=lib_values, default=[], key=f"lib_sel_{idx}")
                            txt = st.text_input(f"Commentaire bloc {idx+1}", value="", key=f"lib_com_{idx}")
                            comment_blocks.append({"labels": sel, "comment": txt})

        # Mapping -> PF + commentaires
        df_map = map_source_to_pf(
            df_clean,
            default_brand=default_brand,
            comment_mode=comment_mode,
            global_comment=global_comment,
            comment_blocks=comment_blocks,
        )
        _log_dedup(
            "mapping",
            (len(df_map),),  # signature = (lignes mappées)
            f"Mapping: {len(df_map)} lignes"
        )


        # Génération d'Index séquentiels (max+1)
        df_map, gen = assign_indexes_sequential(df_map)

        if gen.get("assigned", 0) > 0:
            _log_dedup(
                "indexes",
                (gen["assigned"], gen.get("last")),  # signature = (nb assignés, dernier code)
                f"Index générés: {gen['assigned']} (dernier: {gen.get('last')})"
            )

            st.caption(f"Nouveaux Index: de {gen.get('first')} à {gen.get('last')} (prefix {gen.get('prefix','PF')}).")


        # ==================== LAYOUT ====================
        col1, col2 = st.columns(2)

        # --------- Aperçu ---------
        with col1:
            with st.container(border=True):
                st.subheader("Aperçu")
                st.dataframe(df_map.head(50), use_container_width=False, hide_index=True)

        # --------- Console ---------
        with col2:
            with st.container(border=True):
                st.subheader("🖥️ Console")
                st.code("\n".join(st.session_state[CONSOLE_KEY][-200:]) or "(vide)")
                col1,col2,col3=st.columns(3)
                with col2 :
                    go = st.button("Mettre à jour la base", type="primary", key="btn_go")


        # ===== Action du bouton =====
        if 'go' not in locals():
            go = False

        if go:
            ts0 = _ts_now()
            try:
                # Upsert + audit
                save_products(Kind.PF, df_map)
                _reload_tables_from_sql()

                # Audit & recap minimal
                audit_df = _fetch_audit_since(ts0)
                groups = _group_changes(audit_df)
                created = [g for g in groups if g.get("created")]
                updated = [g for g in groups if (not g.get("created")) and g.get("fields")]

                st.toast("Base mise à jour", icon="✅")

            except Exception as e:
                st.error(f"Échec de la mise à jour : {type(e).__name__}: {e}")



with tab_audit_et_sauvegarde:

    # =================== UI ===================
    actions = list_actions()

    containertop=st.container(border=True)

    with containertop:
        action_sel = st.selectbox(
            "Type d'action",
            ["(toutes)"] + actions,
            index=(["(toutes)"] + actions).index("ELIMINATION_DOUBLONS")
                if "ELIMINATION_DOUBLONS" in actions else 0
        )

    where = "" if action_sel == "(toutes)" else "WHERE action = ?"
    params = [] if where == "" else [action_sel]

    with get_conn() as conn:
        df_audit = pd.read_sql_query(f"""
            SELECT id, ts, user, action, table_name, rec_key, note
            FROM audit_log
            {where}
            ORDER BY ts DESC
        """, conn, params=params)


    st.session_state.setdefault("journal_confirm_reset", False)
    st.session_state.setdefault("journal_pending", {})   # { f"del_{i}":True / f"undo_{i}":True }

    with containertop :
        top_l, top_r = st.columns([4,2])
        with top_l:
            all_users = ["Tous"] + list_known_users()
            user_choice = st.selectbox("Utilisateur", options=all_users, index=0)
            picked_user = None if user_choice == "Tous" else user_choice

            c1, c2 = st.columns(2)
            dfrom = c1.date_input("Du", value=None)
            dto   = c2.date_input("Au", value=None)

        with top_r:
            st.markdown("")
            st.markdown("")
            if not st.session_state["journal_confirm_reset"]:
                if st.button("🧹 Vider le journal", use_container_width=True, type="secondary", key="wipe1"):
                    st.session_state["journal_confirm_reset"] = True
                    st.rerun()
            else:
                cw1, cw2 = st.columns([1,1])
                with cw1:
                    if st.button("✅ Confirmer", use_container_width=True, key="wipe_confirm"):
                        try:
                            reset_audit_log()
                            st.cache_data.clear()
                            st.success("Journal vidé ✅")
                            st.session_state["journal_confirm_reset"] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Réinitialisation échouée : {e}")
                with cw2:
                    if st.button("↩️ Annuler", use_container_width=True, key="wipe_cancel"):
                        st.session_state["journal_confirm_reset"] = False
                        st.rerun()
            st.markdown("")
            if st.button("↻ Rafraîchir", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

    df = load_cell_audit(
        user=picked_user if picked_user else None,
        dfrom=(dfrom.isoformat() if isinstance(dfrom, date) else None),
        dto=(dto.isoformat() if isinstance(dto, date) else None),
        limit=1000
    )


    if df.empty:
        st.info("Aucune modification à afficher.")
    else:
        groups = _build_groups(df)

        for i, g in enumerate(groups):

            action = g["action"]
            table  = g.get("table", "products")

            # Détection des blocs multiples
            is_bulk_create = (action == "CREATE_BULK")
            is_bulk_delete = (action == "DELETE_BULK")
            is_bulk = is_bulk_create or is_bulk_delete


            pill_cls, pill_txt = _pill_for_action(action, g.get("count"))

                
            # clés de confirmation
            k_del = f"del_{i}"
            k_undo = f"undo_{i}"
            pend = st.session_state["journal_pending"]

            with st.container(border=True):
                # En-tête : badge (gauche) + Titre (centre) + actions (droite)
                h1, h2, h3 = st.columns([1.4, 6.3, 2.3])

                with h1:
                    st.markdown(f"<span class='pill {pill_cls}'>{pill_txt}</span>", unsafe_allow_html=True)

                with h2:
                    if is_bulk:
                        header_html = (
                            f"**Création/Suppression multiple** "
                            f"· <span class='badge'>{g['user']}</span>"
                        )
                        ts_display = g.get("ts_sec")
                        st.markdown(header_html, unsafe_allow_html=True)
                        st.markdown(f"<span class='meta'>{g['user']} · {ts_display}</span>", unsafe_allow_html=True)
                    else:
                        if table == "products":
                            header_html = (
                                f"**Index** <span class='code'>{g['rec_key']}</span> "
                                f"· **Libellé** <span class='badge'>{g.get('label','—')}</span>"
                            )
                        elif table == "fournisseurs":
                            sup_name = _supplier_name(g.get("rec_key"), g.get("before_json"), g.get("after_json"))
                            header_html = f"**Fournisseur** · <span class='badge'>{sup_name}</span>"
                        else:  # bom_edges
                            parent_ix, child_ix = (g.get('parent_index',''), g.get('child_index',''))
                            parent_label, child_label = _edge_labels(parent_ix, child_ix)
                            header_html = (
                                f"**Lien** <span class='code'>{parent_ix} → {child_ix}</span> "
                                f"· <span class='badge'>{parent_label}</span> → <span class='badge'>{child_label}</span>"
                            )

                        st.markdown(header_html, unsafe_allow_html=True)
                        st.markdown(f"<span class='meta'>{g['user']} · {g['ts_sec']}</span>", unsafe_allow_html=True)

                with h3:
                    cdel, cundo, cconfirm = st.columns([1,1,1.2])

                    # supprimer bloc (2 étapes)
                    if not pend.get(k_del, False):
                        if cdel.button("🗑️", key=f"delgrp_{i}", help="Supprimer ce bloc du journal", use_container_width=True):
                            pend[k_del] = True
                            pend.pop(k_undo, None)
                            st.rerun()
                    else:
                        if cconfirm.button("✅ Confirmer", key=f"confirm_del_{i}", use_container_width=True):
                            try:
                                delete_audit_ids(g["ids"])
                                st.cache_data.clear()
                                st.success("Bloc supprimé du journal ✅")
                                pend.pop(k_del, None)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Suppression échouée : {e}")
                        if cundo.button("↩️ Annuler", key=f"cancel_del_{i}", use_container_width=True):
                            pend.pop(k_del, None)
                            st.rerun()

                    # UNDO bloc (2 étapes)
                    if not pend.get(k_undo, False):
                        if cundo.button("↩️", key=f"undogrp_{i}", help="Revenir en arrière (annuler ce bloc)", use_container_width=True):
                            pend[k_undo] = True
                            pend.pop(k_del, None)
                            st.rerun()
                    else:
                        if cconfirm.button("✅ Confirmer", key=f"confirm_undo_{i}", use_container_width=True):
                            ok, msg = revert_group(g)
                            if ok:
                                try:
                                    delete_audit_ids(g["ids"])
                                except Exception:
                                    pass
                                st.cache_data.clear()
                                st.success(msg)
                            else:
                                st.error(msg)
                            pend.pop(k_undo, None)
                            st.rerun()
                        if cdel.button("✖️", key=f"cancel_undo_{i}", help="Annuler l'opération", use_container_width=True):
                            pend.pop(k_undo, None)
                            st.rerun()

                # Détails
                if is_bulk:
                    # Dépliant avec détails des enfants
                    with st.expander(f"Détails ({g.get('count', 0)} éléments)"):
                        for j, child in enumerate(g.get("children", []), 1):
                            if child.get("table") == "products":
                                head = f"#{j} — Index {child['rec_key']} · {child.get('label','—')} · {child['ts_sec']}"
                            elif child.get("table") == "fournisseurs":
                                sup_name = _supplier_name(child.get("rec_key"), child.get("before_json"), child.get("after_json"))
                                head = f"#{j} — Fournisseur {sup_name} · {child['ts_sec']}"
                            else:  # bom_edges
                                parent_ix, child_ix = (child.get('parent_index',''), child.get('child_index',''))
                                head = f"#{j} — Lien {parent_ix} → {child_ix} · {child.get('label','—')} · {child['ts_sec']}"

                            st.markdown(f"<div class='childline'><b>{head}</b></div>", unsafe_allow_html=True)

                            chs = child.get("changes", []) or []
                            if chs:
                                ncols = _ncols_for_changes(len(chs))
                                cols = st.columns(ncols)
                                for k, ch in enumerate(chs):
                                    col = cols[k % ncols]
                                    with col:
                                        st.markdown(
                                            f"<div class='diffitem'>"
                                            f"<div><b>{ch.get('field','')}</b></div>"
                                            f"<div><span class='code'>{ch.get('old','')}</span>"
                                            f"<span class='arrow'>→</span>"
                                            f"<span class='code'>{ch.get('new','')}</span></div>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
                else:
                    changes = g.get("changes", []) or []
                    if changes:
                        ncols = _ncols_for_changes(len(changes))
                        cols = st.columns(ncols)
                        for j, ch in enumerate(changes):
                            col = cols[j % ncols]
                            with col:
                                st.markdown(
                                    f"<div class='diffitem'>"
                                    f"<div><b>{ch.get('field','')}</b></div>"
                                    f"<div><span class='code'>{ch.get('old','')}</span>"
                                    f"<span class='arrow'>→</span>"
                                    f"<span class='code'>{ch.get('new','')}</span></div>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )

# ── Barre latérale (à la fin du fichier) ──────────────────────────────────────
with st.sidebar:

    st.markdown("### Session")

    # Affiche l'utilisateur courant si dispo
    user = st.session_state.get("current_user") or "inconnu"
    st.caption(f"Connecté : **{user}**")

    # Bouton déconnexion
    if st.button("Se déconnecter", use_container_width=True):
        # Option 1 (recommandée) : ne réinitialiser que l’auth
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.is_logged_in = False

        # (Option 2 si tu veux tout réinitialiser: st.session_state.clear())
        # st.session_state.clear()

        st.switch_page("Connexion.py")

    st.divider()


    if st.button("🔐 Aller à la connexion", use_container_width=True):
        st.switch_page("Connexion.py")
    if st.button("📦 Aller à l’application", use_container_width=True):
        st.switch_page("Application.py")


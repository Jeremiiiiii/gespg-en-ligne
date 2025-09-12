#Application
import streamlit as st
import pandas as pd
import time, uuid
from typing import List, Dict, Any, Tuple
from streamlit_echarts import st_echarts

from services.donnees import (
    load_products,
    replace_session_df_in_place,
    build_index_map_normalized, 
    ensure_index_map_in_state, 
    update_index_map_for_state,
    _normalize_index_value,
    _reload_tables_from_sql,
    _bump_aggrid_refresh,

    Kind
)

from services.interaction_DB import (
    get_children_of_parent,
    add_child_to_parent_v2,
    remove_child_from_parent_v2,
    calculer_prix_sql,
    _compute_pr_str,
    save_products,
    get_children_info,
    get_levels
)

@st.dialog("Fiche produit", width="large")
def fiche_produit_dialog(selected: List[Dict]) -> None:

    if not selected:
        st.info("Aucune ligne sélectionnée.")
        return

    current = selected[0] if isinstance(selected, (list, tuple)) else (selected if isinstance(selected, dict) else {})
    current_map = dict(current)

    # Heuristique: présence de colonnes d'achat/fournisseur => Semi
    is_semi = ("PA" in current_map) or ("Fournisseur" in current_map) or ("Désignation fournisseur" in current_map)
    source_type = "Semi" if is_semi else "Fini"

    if source_type == "Fini":
        df_target = st.session_state.df_data
        kind = Kind.PF
    else:
        df_target = st.session_state.df_data2
        kind = Kind.SF

    if "Prix d'achat" not in current_map and "PA" in current_map:
        current_map["Prix d'achat"] = current_map.get("PA", "")
    if "PA" not in current_map and "Prix d'achat" in current_map:
        current_map["PA"] = current_map.get("Prix d'achat", "")

    idx_orig = _normalize_index_value(current_map.get("Index", ""))

    def semi_finished_compositions():
        df2 = st.session_state.get("df_data2", pd.DataFrame())
        if not isinstance(df2, pd.DataFrame) or df2.empty:
            return [""], {}
        comp_ser = df2["Composition"].astype(str).fillna("").str.strip() if "Composition" in df2.columns else pd.Series([""] * len(df2), index=df2.index, dtype=str)
        nonempty = comp_ser[comp_ser != ""].tolist()
        seen, unique_comps = set(), []
        for c in nonempty:
            if c not in seen:
                seen.add(c)
                unique_comps.append(c)
        labels = [""] + unique_comps
        rows = df2.to_dict(orient="records")
        mapping = {}
        for comp, row in zip(comp_ser.tolist(), rows):
            if comp and comp not in mapping:
                mapping[comp] = row
        return labels, mapping

    def build_options(col: str):
        frames = []
        df_full  = st.session_state.get("df_full")
        df2_full = st.session_state.get("df2_full")
        df_tgt   = df_target

        if isinstance(df_full, pd.DataFrame) and col in df_full.columns:
            frames.append(df_full[[col]].astype(str))
        if isinstance(df2_full, pd.DataFrame) and col in df2_full.columns:
            frames.append(df2_full[[col]].astype(str))
        if isinstance(df_tgt, pd.DataFrame) and col in df_tgt.columns:
            frames.append(df_tgt[[col]].astype(str))

        if not frames:
            return [""]

        ser = pd.concat(frames, ignore_index=True)[col].fillna("").str.strip()
        ser = ser[ser != ""]
        return [""] + pd.Series(ser.unique()).tolist()


    def ensure_current_in_options(options: List[str], current_value: str) -> List[str]:
        cur = (str(current_value) or "").strip()
        if cur and cur not in options:
            return [cur] + options
        return options

    st.markdown(
        """
        <style>
        .fpd-header{padding:10px 0 6px 0;border-bottom:1px solid #efefef;margin-bottom:12px;}
        .fpd-name{font-size:1.25rem;font-weight:600;line-height:1.2;margin:0;}
        .fpd-type{font-size:.95rem;color:#6b7280;margin-top:2px;}
        .fpd-section{margin-top:12px;}
        .fpd-section-title{font-size:.95rem;font-weight:600;color:#374151;margin-bottom:8px}
        .fpd-card{padding:14px;border:1px solid #ececec;border-radius:12px;background:#fff;margin-bottom:10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    tab_information, tab_composants, tab_coûts = st.tabs(["Information", "Composants","Coûts"])

    # ================== TAB INFORMATION ==================
    with tab_information:
        lib = current_map.get("Libellé produit", current_map.get("Désignation fournisseur", "Produit"))
        st.markdown(
            f"""
            <div class="fpd-header">
                <div class="fpd-name">{(lib or 'Produit')}</div>
                <div class="fpd-type">{'Produit semi-fini' if source_type=='Semi' else 'Produit fini'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        containertitreficheproduit1 = st.container(border=True)
        containertitreficheproduit2 = st.container(border=True)

        with st.form(key=f"form_{source_type}_{idx_orig}", border=False):

            new_values: Dict[str, Any] = {}

            with containertitreficheproduit1:
                st.markdown('''
                <div class="fpd-section" style="
                    display: flex; justify-content: center; align-items: center; height: 40px;
                    border: 1px solid #ccc; padding-top: 17px; padding-bottom: 10px; margin-bottom: 25px;
                    border-radius: 5px; background-color: #f3f4f6; color: #111827; border-color: #e5e7eb;
                    box-shadow: 0 1px 1px rgba(0,0,0,.04), 0 2px 6px rgba(0,0,0,.06);
                "><div class="fpd-section-title" style="text-align: center;">CATÉGORIE</div></div>
                ''', unsafe_allow_html=True)

                containercompo = st.container(border=False)
                c1, c2, c3 = st.columns(3)
                d1, d2, d3 = st.columns(3)

                container_compo_semifini = st.container()
                t1, t2, t3 = st.columns(3)
                z1, z2 = st.columns(2)
                w1, w2, w3 = st.columns(3)

            with containertitreficheproduit2:
                st.markdown('''
                <div class="fpd-section" style="
                    display: flex; justify-content: center; align-items: center; height: 40px;
                    border: 1px solid #ccc; padding-top: 17px; padding-bottom: 10px; margin-bottom: 25px;
                    border-radius: 5px; background-color: #f3f4f6; color: #111827; border-color: #e5e7eb;
                    box-shadow: 0 1px 1px rgba(0,0,0,.04), 0 2px 6px rgba(0,0,0,.06);
                "><div class="fpd-section-title" style="text-align: center;">DÉTAILS & TARIFS</div></div>
                ''', unsafe_allow_html=True)

                f1, f2 = st.columns([3, 1])
                g1, g2, g3 = st.columns(3)
                containercommentaire = st.container()

                h1, h2 = st.columns([3, 1])
                u1, u2, u3 = st.columns(3)
                container_commentaire_semifini = st.container()

            if source_type == "Fini":
                with containercompo:
                    opts_comp, comp_map = semi_finished_compositions()
                    cur_raw = str(current_map.get("Composition", "")).strip()
                    opts_comp = ensure_current_in_options(opts_comp, cur_raw)
                    new_values["Composition"] = st.selectbox(
                        "Composition (sélectionner une composition trouvée dans les semi-finis)",
                        opts_comp,
                        index=opts_comp.index(cur_raw) if cur_raw in opts_comp else 0,
                        help="Choisir une composition déjà utilisée par un semi-fini (aucune saisie libre).",
                        key=f"comp_{idx_orig}",
                    )
                    st.session_state["fpd_selected_semi"] = comp_map.get(new_values["Composition"], {})

                opts_couleur  = ensure_current_in_options(build_options("Couleur"),  str(current_map.get("Couleur", "")).strip())
                opts_marque   = ensure_current_in_options(build_options("Marque"),   str(current_map.get("Marque", "")).strip())
                opts_famille  = ensure_current_in_options(build_options("Famille"),  str(current_map.get("Famille", "")).strip())
                opts_libfam   = ensure_current_in_options(build_options("Libellé famille"), str(current_map.get("Libellé famille", "")).strip())
                opts_unite    = ensure_current_in_options(build_options("Unité"),    str(current_map.get("Unité", "")).strip())
                opts_code     = ensure_current_in_options(build_options("Code liaison externe"), str(current_map.get("Code liaison externe", "")).strip())

                new_values["Couleur"] = c1.selectbox("Couleur", opts_couleur, index=opts_couleur.index(str(current_map.get("Couleur", "")).strip()) if str(current_map.get("Couleur", "")).strip() in opts_couleur else 0, key=f"sb_couleur_{idx_orig}")
                new_values["Marque"]  = c2.selectbox("Marque",  opts_marque,  index=opts_marque.index(str(current_map.get("Marque", "")).strip())  if str(current_map.get("Marque", "")).strip()  in opts_marque  else 0, key=f"sb_marque_{idx_orig}")
                new_values["Famille"] = c3.selectbox("Famille", opts_famille, index=opts_famille.index(str(current_map.get("Famille", "")).strip()) if str(current_map.get("Famille", "")).strip() in opts_famille else 0, key=f"sb_famille_{idx_orig}")

                new_values["Libellé famille"]      = d1.selectbox("Libellé famille",      opts_libfam, index=opts_libfam.index(str(current_map.get("Libellé famille", "")).strip())      if str(current_map.get("Libellé famille", "")).strip()      in opts_libfam else 0, key=f"sb_libfam_{idx_orig}")
                new_values["Unité"]                = d2.selectbox("Unité",                 opts_unite,  index=opts_unite.index(str(current_map.get("Unité", "")).strip())                 if str(current_map.get("Unité", "")).strip()                 in opts_unite  else 0, key=f"sb_unite_{idx_orig}")
                new_values["Code liaison externe"] = d3.selectbox("Code liaison externe",  opts_code,   index=opts_code.index(str(current_map.get("Code liaison externe", "")).strip())   if str(current_map.get("Code liaison externe", "")).strip() in opts_code else 0, key=f"sb_code_{idx_orig}")

            else:
                with containertitreficheproduit2:
                    opts_unite_semi = ensure_current_in_options(build_options("Unité"), str(current_map.get("Unité", "")).strip())
                    new_values["Unité"] = w2.selectbox("Unité", opts_unite_semi, index=opts_unite_semi.index(str(current_map.get("Unité", "")).strip()) if str(current_map.get("Unité", "")).strip() in opts_unite_semi else 0, key=f"sb_unite_semi_{idx_orig}")

            if source_type == "Fini":
                new_values["Libellé produit"] = f1.text_input("Libellé produit", str(current_map.get("Libellé produit", "")).strip(), key=f"ti_libelle_{idx_orig}")
                new_values["Référence"]       = f2.text_input("Référence",       str(current_map.get("Référence", "")).strip(),       key=f"ti_ref_{idx_orig}")
                new_values["Prix d'achat"]    = g1.text_input("Prix d'achat",    str(current_map.get("Prix d'achat", "")).strip(),    key=f"ti_pa_{idx_orig}")
                
                pr_calc = _compute_pr_str(idx_orig)
                g2.text_input("PR (calculé)", pr_calc, key=f"ti_pr_{idx_orig}_disabled", disabled=True, help="Calcul automatique (non modifiable)")

                new_values["PV TTC"]          = g3.text_input("PV TTC",          str(current_map.get("PV TTC", "")).strip(),          key=f"ti_pv_{idx_orig}")
                with containercommentaire:
                    new_values["Commentaire"] = st.text_area("Commentaire", str(current_map.get("Commentaire", "")).strip(), key=f"ta_comment_{idx_orig}")

            else:
                new_values["Composition"] = container_compo_semifini.text_input("Composition", str(current_map.get("Composition", "")).strip(), key=f"ti_comp_{idx_orig}")
                new_values["Libellé produit"] = h1.text_input("Libellé produit", str(current_map.get("Libellé produit", "")).strip(), key=f"ti_libelle_{idx_orig}")
                new_values["Référence"]   = h2.text_input("Référence",   str(current_map.get("Référence", "")).strip(),   key=f"ti_ref_{idx_orig}")
                new_values["Couleur"]     = t1.text_input("Couleur",     str(current_map.get("Couleur", "")).strip(),     key=f"ti_couleur_{idx_orig}")
                new_values["Marque"]      = t2.text_input("Marque",      str(current_map.get("Marque", "")).strip(),      key=f"ti_marque_{idx_orig}")
                new_values["Famille"]     = t3.text_input("Famille",     str(current_map.get("Famille", "")).strip(),     key=f"ti_famille_{idx_orig}")
                new_values["Fournisseur"] = z1.text_input("Fournisseur", str(current_map.get("Fournisseur", "")).strip(), key=f"ti_fournisseur_{idx_orig}")
                new_values["Désignation fournisseur"] = z2.text_input("Désignation fournisseur", str(current_map.get("Désignation fournisseur", "")).strip(), key=f"ti_desfourn_{idx_orig}")
                new_values["Libellé famille"]      = w1.text_input("Libellé famille",      str(current_map.get("Libellé famille", "")).strip(),      key=f"ti_libfam_{idx_orig}")
                new_values["Code liaison externe"] = w3.text_input("Code liaison externe", str(current_map.get("Code liaison externe", "")).strip(), key=f"ti_code_{idx_orig}")
                new_values["Prix d'achat"] = u1.text_input("Prix d'achat", str(current_map.get("Prix d'achat", "")).strip(), key=f"ti_pa_{idx_orig}")
                
                pr_calc = _compute_pr_str(idx_orig)
                u2.text_input("PR (calculé)", pr_calc, key=f"ti_pr_{idx_orig}_disabled", disabled=True, help="Calcul automatique (non modifiable)")

                new_values["PV TTC"]       = u3.text_input("PV TTC",       str(current_map.get("PV TTC", "")).strip(),       key=f"ti_pv_{idx_orig}")
                with container_commentaire_semifini:
                    new_values["Commentaire"] = st.text_area("Commentaire", str(current_map.get("Commentaire", "")).strip(), key=f"ta_comment_{idx_orig}")

            save_btn = st.form_submit_button("💾 Enregistrer", use_container_width=True)

        # ---- Sauvegarde CSV + resync SQL.products ----
        if save_btn:
            try:
                target_df = df_target.copy() if isinstance(df_target, pd.DataFrame) else pd.DataFrame()
                if "Index" not in target_df.columns:
                    st.error("Le fichier ne contient pas de colonne 'Index'.")
                    st.stop()

                if not target_df.empty:
                    target_df["Index"] = target_df["Index"].astype(str).map(_normalize_index_value)

                idx_new = idx_orig
                values = {**current_map, **new_values}
                values["Index"] = idx_new
                if source_type == "Semi" and "Prix d'achat" in new_values:
                    values["PA"] = new_values["Prix d'achat"]

                mask = (target_df["Index"].astype(str).map(_normalize_index_value) == idx_orig)
                if mask.any():
                    for k, v in values.items():
                        if k in target_df.columns:
                            target_df.loc[mask, k] = v
                else:
                    new_row = {c: "" for c in target_df.columns}
                    for k, v in values.items():
                        if k in new_row:
                            new_row[k] = v
                    target_df = pd.concat([target_df, pd.DataFrame([new_row])], ignore_index=True)

                save_products(kind, target_df)   # upsert vers SQL
                _reload_tables_from_sql()        # recharger depuis SQL



                st.toast("Enregistrement fini", icon="✅")
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement : {e}")

    # ================== TAB COMPOSANTS ==================
    with tab_composants:
        st.markdown(
            f"""
            <div class="fpd-header">
                <div class="fpd-name">{(current_map.get("Libellé produit", current_map.get("Désignation fournisseur","Produit")))}</div>
                <div class="fpd-type">{'Arbre de parenté & Quantités'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if source_type == "Fini":
            parent_state_key = "df_data"     # parent = produits finis
            component_state_key = "df_data2" # enfants = semi-finis
        else:
            parent_state_key = "df_data2"    # parent = semi-fini
            component_state_key = "df_data2" # enfants = semi-finis

        ensure_index_map_in_state(parent_state_key)
        if component_state_key:
            ensure_index_map_in_state(component_state_key)

        children = get_children_of_parent(idx_orig, parent_state_key, component_state_key)

        if not children:
            st.info("Aucun enfant lié à ce produit.")
        else:
            container_titre_composants_FP = st.container(border=True)
            with container_titre_composants_FP:
                Index_composants_FP, Libellé_composants_FP, Qté_composants_FP, Unité_composants_FP, Supprimer_composants_FP = st.columns([2, 6, 3, 1.2, 2])
                Index_composants_FP.markdown("<divcomposants style='display:flex;align-items:center;justify-content:center;font-weight:600;padding-bottom:3px'>Index</divcomposants>", unsafe_allow_html=True)
                Libellé_composants_FP.markdown("<divcomposants style='display:flex;align-items:center;justify-content:center;font-weight:600;padding-bottom:3px'>Libellé</divcomposants>", unsafe_allow_html=True)
                Qté_composants_FP.markdown("<divcomposants style='display:flex;align-items:center;justify-content:center;font-weight:600;padding-bottom:3px'>Quantité</divcomposants>", unsafe_allow_html=True)
                Unité_composants_FP.markdown("<divcomposants style='display:flex;align-items:center;justify-content:center;font-weight:600;padding-bottom:3px'>Unité</divcomposants>", unsafe_allow_html=True)
                Supprimer_composants_FP.markdown("<divcomposants style='display:flex;align-items:center;justify-content:center;font-weight:600;padding-bottom:3px'>Bouton</divcomposants>", unsafe_allow_html=True)

            for child in children:
                row_container = st.container(border=True)
                with row_container:
                    cidx = _normalize_index_value(child.get("Index", ""))
                    lib_enfant = child.get("Libellé produit", "") or child.get("Désignation fournisseur", "")
                    qty = child.get("Quantité_pour_parent", "")
                    unit = child.get("Unité", "") or child.get("Unite", "") or ""
                    missing = bool(child.get("missing", False))

                    row_ph = st.empty()
                    with row_ph.container():
                        cols = st.columns([2, 6, 3, 1.2, 2])
                        cols[0].markdown(f"<div style='display:flex;align-items:center;min-height:130px;justify-content:center;font-weight:400;'>{cidx}</div>", unsafe_allow_html=True)
                        cols[1].markdown(f"<div style='display:flex;width:100%;align-items:center;min-height:130px;justify-content:center;font-weight:400;'>{lib_enfant}{' — (manquant)' if missing else ''}</div>", unsafe_allow_html=True)

                        form_key_upd = f"form_update_{idx_orig}_{cidx}"
                        with cols[2].form(key=form_key_upd, clear_on_submit=False):
                            new_qty = st.text_input("", value=str(qty or ""), key=f"qty_inp_{idx_orig}_{cidx}")
                            submit_update = st.form_submit_button("💾")
                            if submit_update:
                                st.session_state["fpd_keep_open"] = True
                                res = add_child_to_parent_v2(
                                    parent_state_key=parent_state_key,
                                    component_state_key=component_state_key,
                                    parent_index=idx_orig, child_index=cidx,
                                    quantite=str(new_qty) if new_qty is not None else "0",
                                    rerun_after=False, clear_cache_after_write=True
                                )


                                if not res.get("ok"):
                                    st.error(res.get("msg"))
                                else:
                                    placeholder = st.empty()
                                    placeholder.success("✅")
                                    time.sleep(1.2)
                                    placeholder.empty()
                                ensure_index_map_in_state(parent_state_key)
                                if component_state_key:
                                    ensure_index_map_in_state(component_state_key)

                        cols[3].markdown(f"<div style='display:flex;align-items:center;min-height:130px;justify-content:center;font-weight:400;'>{unit}</div>", unsafe_allow_html=True)

                        form_key_del = f"form_delete_{idx_orig}_{cidx}"
                        with cols[4]:
                            st.markdown("<div style='display:flex;align-items:center;justify-content:center;min-height:45px;'>", unsafe_allow_html=True)
                            with st.form(key=form_key_del, clear_on_submit=False, border=False):
                                submit_del = st.form_submit_button("🗑️")
                            st.markdown("</div>", unsafe_allow_html=True)

                            if submit_del:
                                st.session_state["fpd_keep_open"] = True
                                res = remove_child_from_parent_v2(
                                    parent_state_key=parent_state_key,          # "df_data" si PF, sinon "df_data2"
                                    component_state_key=component_state_key,    # "df_data2" dans les deux cas
                                    parent_index=idx_orig,                      # index du produit affiché
                                    child_index=cidx,                           # index de l’enfant à supprimer
                                    rerun_after=False,
                                    clear_cache_after_write=True
                                )

                                if not res.get("ok"):
                                    st.error(res.get("msg"))
                                else:
                                    # rafraîchis les maps pour l’UI
                                    ensure_index_map_in_state(parent_state_key)
                                    if component_state_key:
                                        ensure_index_map_in_state(component_state_key)
                                    placeholder = st.empty()
                                    placeholder.success("✅")
                                    time.sleep(1.0)
                                    placeholder.empty()
                                    row_ph.empty()  # retire la ligne visuellement


        st.markdown("### Ajouter un enfant")

        # liste des semi-finis (enfants) possibles depuis df_comp
        df_comp = st.session_state.get(component_state_key) if component_state_key else None
        options = [""]

        if isinstance(df_comp, pd.DataFrame) and "Index" in df_comp.columns:
            df_comp["Index"] = df_comp["Index"].astype(str).apply(_normalize_index_value)
            rows = df_comp.to_dict(orient="records")
            for r in rows:
                ix = _normalize_index_value(r.get("Index", ""))
                if not ix:
                    continue
                label = f"{ix} — {r.get('Libellé produit','')}".strip(" — ")
                options.append(label)

        existing_idx = [_normalize_index_value(c.get("Index", "")) for c in children if c.get("Index")]
        options_filtered = [""] + [opt for opt in options[1:] if opt.split(" — ")[0] not in existing_idx]

        btn_key = f"button_clicked_{idx_orig}"
        added_key = f"added_ok_{idx_orig}"
        st.session_state.setdefault(btn_key, False)
        st.session_state.setdefault(added_key, False)

        form_key_add = f"form_add_{idx_orig}_new"

        def on_add_click(parent_state_key=parent_state_key,
                        component_state_key=component_state_key,
                        parent_index=idx_orig):
            
            st.session_state[btn_key] = not st.session_state[btn_key]
            sel = st.session_state.get(f"sel_child_{parent_index}_new")
            qty = st.session_state.get(f"add_qty_{parent_index}_new", "1")
            st.session_state[added_key] = False

            if sel:
                chosen_idx = str(sel).split(" — ")[0]
                res = add_child_to_parent_v2(
                    parent_state_key=parent_state_key,
                    component_state_key=component_state_key,
                    parent_index=parent_index,
                    child_index=chosen_idx,
                    quantite=str(qty) if qty is not None else "1",
                    rerun_after=False, clear_cache_after_write=True
                )
                st.session_state[added_key] = bool(res.get("ok"))

            ensure_index_map_in_state(parent_state_key)
            if component_state_key:
                ensure_index_map_in_state(component_state_key)


        with st.form(key=form_key_add, clear_on_submit=False):
            sel = st.selectbox("Choisir un semi-fini à ajouter", options_filtered, key=f"sel_child_{idx_orig}_new")
            qty = st.text_input("Quantité pour ce parent", value="1", key=f"add_qty_{idx_orig}_new")
            st.form_submit_button(label="Ajouter enfant 🧑‍🧒‍🧒", on_click=on_add_click)

        if st.session_state.get(added_key):
            placeholder = st.empty()
            placeholder.success("Enfant ajouté ✅")
            time.sleep(1.2)
            placeholder.empty()
            st.session_state[added_key] = False

    # ================== TAB COÛTS ==================

    res = calculer_prix_sql(idx_orig)
    pr_total = res.get("tree", {}).get("PR", None)
    level = get_children_info(res["tree"], idx_orig)
    levels = get_levels(res["tree"])
    nb_niveaux = max(1, len(levels) - 1)
    liste_etage = list(range(1, nb_niveaux + 1))

    if len(levels) > 1:
        with tab_coûts:
            if pr_total is not None:
                st.metric("PR total (calculé)", f"{pr_total}")
            if not level:
                st.info("Aucun enfant lié à ce produit.")

            st.markdown(
                f"""
                <div class="fpd-header">
                    <div class="fpd-name">{(current_map.get("Libellé produit", current_map.get("Désignation fournisseur","Produit")))}</div>
                    <div class="fpd-type">{'Coûts et Quantités'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            df1 = st.session_state.get("df_data", pd.DataFrame())
            df2 = st.session_state.get("df_data2", pd.DataFrame())
            idx_map1 = build_index_map_normalized(df1) if isinstance(df1, pd.DataFrame) and "Index" in getattr(df1, "columns", []) else {}
            idx_map2 = build_index_map_normalized(df2) if isinstance(df2, pd.DataFrame) and "Index" in getattr(df2, "columns", []) else {}

            with col1:
                chart_placeholder = st.empty()

            def pie(dico_index: Dict[str, Dict[str, Any]]) -> None:
                DATA_GRAPH_PIE = [["product", "value"]]
                for index, valeurs in dico_index.items():
                    idx_n = _normalize_index_value(index)
                    label = idx_n
                    if idx_n in idx_map1 and "Libellé produit" in getattr(df1, "columns", []):
                        label = str(df1.iloc[idx_map1[idx_n]]["Libellé produit"]).strip() or idx_n
                    elif idx_n in idx_map2 and "Libellé produit" in getattr(df2, "columns", []):
                        label = str(df2.iloc[idx_map2[idx_n]]["Libellé produit"]).strip() or idx_n

                    qpp = valeurs.get("Quantité_pour_parent") or 0
                    pr  = valeurs.get("PR") or 0
                    try:
                        val = float(qpp) * float(pr)
                    except Exception:
                        val = 0.0
                    DATA_GRAPH_PIE.append([label, val])

                n_items = len(DATA_GRAPH_PIE) - 1
                many = n_items > 8  # ajuste le seuil si besoin

                # Légende en haut, horizontale, avec scroll si trop d’éléments
                legend_opts = {
                    "type": "scroll",
                    "orient": "horizontal",
                    "top": 0,
                    "left": "center",
                    "itemGap": 8,         # espace entre éléments de légende
                    "itemWidth": 12,
                    "itemHeight": 12,
                    # (optionnel) tronquer les libellés trop longs
                    # "formatter": "function(name){return name.length>24 ? name.slice(0,24)+'…' : name;}"
                }

                # On "descend" le camembert pour laisser la légende respirer
                # -> centre vertical plus bas (ex: 60-65%) et un peu plus de hauteur totale
                center_y = "62%" if many else "54%"
                radius   = ["40%", "70%"] if many else ["40%", "60%"]

                option = {
                    "tooltip": {"trigger": "item", "confine": True},
                    "legend": legend_opts,
                    "series": [
                        {
                            "type": "pie",
                            "radius": radius,
                            "center": ["50%", center_y],   # ⬅️ uniquement vertical
                            "avoidLabelOverlap": True,
                            "labelLayout": {"hideOverlap": True},
                            "minShowLabelAngle": 10,
                            "label": {"formatter": "{b}: {c} ({d}%)"},
                            "encode": {"itemName": "product", "value": "value", "tooltip": "value"},
                            "data": [{"name": row[0], "value": row[1]} for row in DATA_GRAPH_PIE[1:]],
                        }
                    ],
                }

                # Plus d'espace vertical s'il y a beaucoup d'éléments dans la légende
                base_h = 320
                extra  = 100 if many else 0
                height_px = base_h + extra

                with chart_placeholder:
                    st_echarts(option, height=f"{height_px}px", key=f"echarts_{idx_orig}_{uuid.uuid4()}")


            pie(level)

            with col2:
                with st.container(border=True):
                    submit = False          # garde-fou si le form n'est pas rendu
                    niveau_i = None         # idem

                    # niveaux descendants (hors racine)
                    n_levels = max(0, len(levels) - 1)

                    if n_levels <= 1:
                        # 0/1 niveau => pas de sélecteur (sinon min==max côté front)
                        st.caption("Un seul niveau d'enfants (n-1).")
                        choix = 1  # fixe
                    else:
                        liste_etage = list(range(1, n_levels + 1))

                        # Préfère un selectbox (tolère 1+ éléments) ou un select_slider si tu y tiens
                        choix = st.selectbox(
                            "Sélection niveau n-",
                            options=liste_etage,
                            index=len(liste_etage) - 1,
                            key=f"slider_{idx_orig}",
                        )

                        with st.form(key=f"form_niveau_{idx_orig}"):
                            if choix is not None:
                                indices_niveau = levels[int(choix) - 1]  # 0=root, 1=n-1, etc.
                                labels_niveau = []
                                label_to_index = {}
                                for idx in indices_niveau:
                                    idx_n = _normalize_index_value(idx)
                                    libelle = idx_n
                                    if idx_n in idx_map1 and "Libellé produit" in getattr(df1, "columns", []):
                                        libelle = str(df1.iloc[idx_map1[idx_n]]["Libellé produit"]).strip() or idx_n
                                    elif idx_n in idx_map2 and "Libellé produit" in getattr(df2, "columns", []):
                                        libelle = str(df2.iloc[idx_map2[idx_n]]["Libellé produit"]).strip() or idx_n
                                    labels_niveau.append(libelle)
                                    label_to_index[libelle] = idx_n

                                if not labels_niveau:
                                    st.info("Aucun élément à ce niveau.")
                                    submit = False
                                    niveau_i = None
                                else:
                                    niveau_i_label = st.selectbox("", labels_niveau, key=f"niveau1{idx_orig}")
                                    niveau_i = label_to_index.get(niveau_i_label, indices_niveau[0])
                                    submit = st.form_submit_button("Valider")

                    if submit:
                        if choix == 1:
                            pie(level)
                        else:
                            if niveau_i:
                                res2 = calculer_prix_sql(niveau_i)
                                level2 = get_children_info(res2["tree"], niveau_i)
                                pie(level2)


#------------------------------------------------------------------------------- FIN/ST.MODAL --------------------------------------------------------------------------------

def _build_options(col: str, df_target: pd.DataFrame) -> List[str]:
    """
    Options uniques pour selectbox à partir de df_full / df2_full / df_target.
    (Lecture pure UI ;  logique parent-enfant en SQL.)
    """
    frames = []

    df_full  = st.session_state.get("df_full")
    df2_full = st.session_state.get("df2_full")


    if isinstance(df_full, pd.DataFrame) and col in getattr(df_full, "columns", []):
        frames.append(df_full[[col]].astype(str))
    if isinstance(df2_full, pd.DataFrame) and col in getattr(df2_full, "columns", []):
        frames.append(df2_full[[col]].astype(str))
    if isinstance(df_target, pd.DataFrame) and col in getattr(df_target, "columns", []):
        frames.append(df_target[[col]].astype(str))

    if not frames:
        return [""]

    ser = pd.concat(frames, ignore_index=True)[col].fillna("").str.strip()
    ser = ser[ser != ""]
    uniq = pd.Series(ser.unique()).tolist()
    return [""] + uniq


def _semi_finished_compositions() -> Tuple[List[str], Dict[str, Any]]:
    """
    Labels & mapping des 'Composition' existantes dans df_data2 (semi-finis).
    (Toujours basé sur CSV car 'Composition' n'est pas dans la base SQL.)
    """
    df2 = st.session_state.get("df_data2")
    if not isinstance(df2, pd.DataFrame) or df2.empty:
        return [""], {}

    if "Composition" in df2.columns:
        comp_ser = df2["Composition"].astype(str).fillna("").str.strip()
    else:
        comp_ser = pd.Series([""] * len(df2), dtype=str)

    nonempty = comp_ser[comp_ser != ""].tolist()
    seen, unique_comps = set(), []
    for c in nonempty:
        if c not in seen:
            seen.add(c)
            unique_comps.append(c)

    labels = [""] + unique_comps

    # première ligne porteuse de la composition
    rows = df2.to_dict(orient="records")
    mapping: Dict[str, Any] = {}
    for comp, row in zip(comp_ser.tolist(), rows):
        if comp and comp not in mapping:
            mapping[comp] = row

    return labels, mapping


# ========================= DIALOG CRÉATION =========================
@st.dialog("Créer une fiche produit", width="large")
def creation_fiche_produit_dialog(source_type: str) -> None:
    """
    Création d'une nouvelle fiche (persistance directe en SQL.products).
    - 'Fini'  => l'Index DOIT commencer par 'PF'
    - 'Semi'  => l'Index NE DOIT PAS commencer par 'PF'
    - Unicité contrôlée via le DataFrame en session (rechargé depuis SQL)
    """
    # --- Normalisation du type -> Kind.PF / Kind.SF ---
    src = str(source_type).strip().lower()
    kind = Kind.SF if src.startswith("semi") else Kind.PF

    if kind == Kind.SF:
        df_key = "df_data2"
        type_label = "Produit semi-fini"
        # (on ne remet PAS ArbreParenté / Visible / Quantité:pour)
        expected_order = [
            "Index","Libellé produit","Composition","Couleur","Unité","Fournisseur","Désignation fournisseur",
            "PA","PR","Code liaison externe","Commentaire","Prix d'achat","Marque","Famille","Référence","PV TTC","Libellé famille"
        ]
    else:
        df_key = "df_data"
        type_label = "Produit fini"
        expected_order = [
            "Index","Référence","Libellé produit","Composition","Couleur","Marque","Famille","Libellé famille",
            "Prix d'achat","PR","Unité","PV TTC","Code liaison externe","Commentaire"
        ]

    # Charger le DF cible depuis SQL si absent en session
    df_target = st.session_state.get(df_key)
    if not isinstance(df_target, pd.DataFrame):
        df_target = load_products(kind)                 # <-- lit depuis SQL.products
        replace_session_df_in_place(df_key, df_target)  # garde l’objet DataFrame identique pour AgGrid
        _bump_aggrid_refresh()
    update_index_map_for_state(df_key)

    # ---------- helper: garantir colonnes & ordre exact (sans columns BOM héritées) ----------
    def _ensure_columns_and_order(df: pd.DataFrame, order: List[str]) -> pd.DataFrame:
        df = df.copy()
        for col in order:
            if col not in df.columns:
                df[col] = ""
        extras = [c for c in df.columns if c not in order]
        return df[order + extras]

    # ---------- Style ----------
    st.markdown(
        """
        <style>
        .fpd-header{padding:10px 0 6px 0;border-bottom:1px solid #efefef;margin-bottom:12px;}
        .fpd-name{font-size:1.25rem;font-weight:600;line-height:1.2;margin:0;}
        .fpd-type{font-size:.95rem;color:#6b7280;margin-top:2px;}
        .fpd-section{margin-top:12px;}
        .fpd-section-title{font-size:.95rem;font-weight:600;color:#374151;margin-bottom:8px}
        .fpd-card{padding:14px;border:1px solid #ececec;border-radius:12px;background:#fff;margin-bottom:10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Onglets ----------
    tab_information, = st.tabs(["Information"])  

    # ================== TAB INFORMATION ==================
    with tab_information:
        st.markdown(
            f"""
            <div class="fpd-header">
                <div class="fpd-name">{"Produit"}</div>
                <div class="fpd-type">{type_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        containertitreficheproduit1 = st.container(border=True)
        containertitreficheproduit2 = st.container(border=True)

        # -------- FORMULAIRE de création --------
        with st.form(key=f"form_create_{source_type}", border=False):
            new_values: Dict[str, Any] = {}
            idx_orig = ""  # création -> vide

            with containertitreficheproduit1:
                st.markdown(
                    '''
                    <div class="fpd-section" style="
                        display:flex;justify-content:center;align-items:center;height:40px;border:1px solid #ccc;
                        padding-top:17px;padding-bottom:10px;margin-bottom:25px;border-radius:5px;background-color:#f3f4f6;
                        color:#111827;border-color:#e5e7eb;box-shadow:0 1px 1px rgba(0,0,0,.04), 0 2px 6px rgba(0,0,0,.06);
                    "><div class="fpd-section-title" style="text-align:center;">CATÉGORIE</div></div>
                    ''',
                    unsafe_allow_html=True
                )
                # Fini
                containercompo = st.container(border=False)
                c1, c2, c3 = st.columns(3)
                d1, d2, d3 = st.columns(3)
                # Semi
                container_compo_semifini = st.container()
                z1, z2 = st.columns(2)
                t1, t2, t3 = st.columns(3)
                w1, w2, w3 = st.columns(3)

            with containertitreficheproduit2:
                st.markdown(
                    '''
                    <div class="fpd-section" style="
                        display:flex;justify-content:center;align-items:center;height:40px;border:1px solid #ccc;
                        padding-top:17px;padding-bottom:10px;margin-bottom:25px;border-radius:5px;background-color:#f3f4f6;
                        color:#111827;border-color:#e5e7eb;box-shadow:0 1px 1px rgba(0,0,0,.04), 0 2px 6px rgba(0,0,0,.06);
                    "><div class="fpd-section-title" style="text-align:center;">DÉTAILS & TARIFS</div></div>
                    ''',
                    unsafe_allow_html=True
                )
                # Index (obligatoire)
                idx_col = st.columns([1])
                new_values["Index"] = idx_col[0].text_input("Index (obligatoire)", value="", key=f"ti_index_{source_type}")
                # Fini
                f1, f2 = st.columns([3, 1])
                g1, g2, g3 = st.columns(3)
                containercommentaire = st.container()
                # Semi
                h1, h2 = st.columns([3, 1])
                u1, u2, u3 = st.columns(3)
                container_commentaire_semifini = st.container()

            # --------- Champs par type ---------
            if source_type == "Fini":
                with containercompo:
                    opts_comp, comp_map = _semi_finished_compositions()
                    new_values["Composition"] = st.selectbox(
                        "Composition (sélectionner une composition trouvée dans les semi-finis)",
                        opts_comp, index=0,
                        help="Choisir une composition déjà utilisée par un semi-fini (aucune saisie libre).",
                        key=f"comp_{source_type}",
                    )
                    st.session_state["fpd_selected_semi"] = comp_map.get(new_values["Composition"], {})

                opts_couleur = _build_options("Couleur", df_target)
                opts_marque  = _build_options("Marque", df_target)
                opts_famille = _build_options("Famille", df_target)

                new_values["Couleur"] = c1.selectbox("Couleur", opts_couleur, index=0, key=f"sb_couleur_{source_type}")
                new_values["Marque"]  = c2.selectbox("Marque",  opts_marque,  index=0, key=f"sb_marque_{source_type}")
                new_values["Famille"] = c3.selectbox("Famille", opts_famille, index=0, key=f"sb_famille_{source_type}")

                opts_libfam = _build_options("Libellé famille", df_target)
                opts_unite  = _build_options("Unité", df_target)
                opts_code   = _build_options("Code liaison externe", df_target)

                new_values["Libellé famille"]      = d1.selectbox("Libellé famille",      opts_libfam, index=0, key=f"sb_libfam_{source_type}")
                new_values["Unité"]                 = d2.selectbox("Unité",                 opts_unite,  index=0, key=f"sb_unite_{source_type}")
                new_values["Code liaison externe"]  = d3.selectbox("Code liaison externe",  opts_code,   index=0, key=f"sb_code_{source_type}")

            else:
                with containertitreficheproduit2:
                    opts_unite_semi = _build_options("Unité", df_target)
                    new_values["Unité"] = w2.selectbox("Unité", opts_unite_semi, index=0, key=f"sb_unite_semi_{source_type}")

            # --------- DÉTAILS & TARIFS ---------
            if source_type == "Fini":
                new_values["Libellé produit"] = f1.text_input("Libellé produit", "", key=f"ti_libelle_{source_type}")
                new_values["Référence"]       = f2.text_input("Référence", "", key=f"ti_ref_{source_type}")
                new_values["Prix d'achat"]    = g1.text_input("Prix d'achat", "", key=f"ti_pa_{source_type}")
                
                # Aperçu non éditable : à la création il n'y a pas encore d'arbo -> PR = éventuel Prix d'achat
                _preview_pr = (new_values.get("Prix d'achat") or "").strip()
                g2.text_input("PR (calculé)", _preview_pr, key=f"ti_pr_{source_type}_disabled", disabled=True, help="Le PR sera recalculé dès que des composants seront liés")

                new_values["PV TTC"]          = g3.text_input("PV TTC", "", key=f"ti_pv_{source_type}")
                with containercommentaire:
                    new_values["Commentaire"] = st.text_area("Commentaire", "", key=f"ta_comment_{source_type}")

            else:
                new_values["Composition"]           = container_compo_semifini.text_input("Composition", "", key=f"ti_comp_{source_type}")
                new_values["Libellé produit"]       = h1.text_input("Libellé produit", "", key=f"ti_libelle_{source_type}")
                new_values["Référence"]             = h2.text_input("Référence", "", key=f"ti_ref_{source_type}")
                new_values["Couleur"]               = t1.text_input("Couleur", "", key=f"ti_couleur_{source_type}")
                new_values["Marque"]                = t2.text_input("Marque",  "", key=f"ti_marque_{source_type}")
                new_values["Famille"]               = t3.text_input("Famille", "", key=f"ti_famille_{source_type}")
                new_values["Libellé famille"]       = w1.text_input("Libellé famille", "", key=f"ti_libfam_{source_type}")
                new_values["Code liaison externe"]  = w3.text_input("Code liaison externe", "", key=f"ti_code_{source_type}")
                new_values["Fournisseur"]           = z1.text_input("Fournisseur","", key=f"ti_fournisseur_{source_type}")
                new_values["Désignation fournisseur"]= z2.text_input("Désignation fournisseur","", key=f"ti_désign_fourn_{source_type}")
                new_values["Prix d'achat"]          = u1.text_input("Prix d'achat", "", key=f"ti_pa_{source_type}")
                
                _preview_pr = (new_values.get("Prix d'achat") or "").strip()
                u2.text_input("PR (calculé)", _preview_pr, key=f"ti_pr_{source_type}_disabled", disabled=True, help="Le PR sera recalculé dès que des composants seront liés")

                new_values["PV TTC"]                = u3.text_input("PV TTC", "", key=f"ti_pv_{source_type}")
                with container_commentaire_semifini:
                    new_values["Commentaire"] = st.text_area("Commentaire", "", key=f"ta_comment_{source_type}")

            save_btn = st.form_submit_button("➕ Créer la fiche", use_container_width=True)

        # ===================== SAUVEGARDE =====================
        if save_btn:
            try:
                # 1) Index requis + normalisation
                idx_new_raw = str(new_values.get("Index", "")).strip()
                if not idx_new_raw:
                    st.error("Veuillez saisir un **Index**.")
                    st.stop()
                idx_new = _normalize_index_value(idx_new_raw)

                # 1bis) Cohérence PF / type
                if source_type == "Fini" and not idx_new.startswith("PF"):
                    st.error("Un **produit fini** doit avoir un Index qui commence par **PF**.")
                    st.stop()
                if source_type == "Semi" and idx_new.startswith("PF"):
                    st.error("Un **semi-fini** ne doit pas avoir un Index qui commence par **PF**.")
                    st.stop()

                # 2) Unicité dans la table cible
                idx_map = st.session_state.get(f"{df_key}__index_map") or {}
                if idx_new in idx_map:
                    st.error(f"L'Index **{idx_new}** existe déjà.")
                    st.stop()

                # 3) Préparer DF cible + colonnes attendues (ordre figé, sans colonnes BOM héritées)
                target_df = df_target.copy() if isinstance(df_target, pd.DataFrame) else pd.DataFrame()
                target_df = _ensure_columns_and_order(target_df, expected_order)

                # 4) Construire la ligne à insérer
                row_dict = {c: "" for c in target_df.columns}
                row_dict.update(new_values)
                row_dict["Index"] = idx_new
                if source_type == "Semi" and "Prix d'achat" in new_values:
                    row_dict["PA"] = new_values["Prix d'achat"]  # miroir PA
                row_df = pd.DataFrame([row_dict]).reindex(columns=target_df.columns, fill_value="")
                target_df = pd.concat([target_df, row_df], ignore_index=True)

                save_products(kind, target_df)   # upsert vers SQL
                _reload_tables_from_sql()        # recharger depuis SQL


                st.toast("Enregistrement effectué, mise à jour du tableau…", icon="🔄")
                time.sleep(0.8)
                _reload_tables_from_sql()  # recharge df_data/df_data2 et synchronise SQL.products
                st.toast("Enregistrement fini, fermeture st.dialog", icon="✅")
                time.sleep(1.2)
                st.rerun()
                

            except Exception as e:

                st.error(f"Erreur lors de l'enregistrement : {e}")

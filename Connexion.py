import streamlit as st
from utils.auth import register_user, authenticate_user 

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("🔐 Connexion")

# --- Init des variables d'état 
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False

# --- Zone Connexion ---
with st.container():
    username = st.text_input("Nom d'utilisateur", key="login_user")
    password = st.text_input("Mot de passe", type="password", key="login_pass")
    if st.button("Se connecter", use_container_width=True):
        if authenticate_user(username, password):
            st.success("Connecté avec succès")
            st.session_state.authenticated = True
            st.session_state.current_user = username
            st.session_state.is_logged_in = True  # compat
            st.switch_page("pages/Application.py")       # ⇦ redirige vers l’app
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect")

st.divider()

# --- Gestion des comptes (visible seulement si admin déjà connecté) ---
if st.session_state.get("authenticated") and st.session_state.get("current_user") == "admin":
    st.subheader("👤 Créer un compte (admin)")
    new_user = st.text_input("Nouveau nom d'utilisateur", key="nu")
    new_pass = st.text_input("Nouveau mot de passe", type="password", key="np")
    if st.button("Créer le compte"):
        if register_user(new_user, new_pass):
            st.success(f"Compte '{new_user}' créé avec succès.")
        else:
            st.warning("Ce nom d'utilisateur existe déjà.")

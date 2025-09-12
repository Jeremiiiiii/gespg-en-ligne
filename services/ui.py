import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

def _inject_css(*names: str):
    for n in names:
        css = Path(f"assets/{n}").read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def _inject_js(name: str):
    js = Path(f"assets/{name}").read_text(encoding="utf-8")
    components.html(f"<script>{js}</script>", height=0)

def use_base_ui(
    page_title: str = "Application",
    sidebar: str = "collapsed",
    inject_scroll_lock: bool = False,   
) -> None:
    st.set_page_config(
        page_title=page_title,
        layout="wide",
        initial_sidebar_state=("collapsed" if sidebar == "collapsed" else "auto"),
    )

    _inject_css("base.css", "tabs.css", "compact.css","audits_et_sauvegardes.css","Trie_donnees.css")
    _inject_js("sidebar_autoclose.js")

    if inject_scroll_lock:
        # on active le lock sur CETTE page
        components.html("""
        <script>
          // cette page autorise le lock global
          window.parent.__ALLOW_SCROLL__ = false;
        </script>
        """, height=0)
        _inject_js("scroll_lock.js")
    else:
        # cette page ne veut PAS de lock → on le coupe si présent
        components.html("""
        <script>
          window.parent.__ALLOW_SCROLL__ = true;              // signal au script de s'abstenir
          if (window.parent.__SCROLL_LOCK__) {
            try { window.parent.__SCROLL_LOCK__.disable(); } catch(e){}
          }
        </script>
        """, height=0)
        # ce CSS neutralise tout overflow:hidden laissé par un lock précédent (ou un CSS global)
        st.markdown("""
        <style>
          html, body { overflow: auto !important; height: auto !important; }
          [data-testid="stAppViewContainer"],
          [data-testid="stMain"],
          .block-container {
            overflow: auto !important;
            height: auto !important;
          }
        </style>
        """, unsafe_allow_html=True)


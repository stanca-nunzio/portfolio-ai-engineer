"""
E-Commerce Sales Dashboard
Pipeline completa: dati sporchi -> pulizia -> analisi statistica -> dashboard interattiva.
"""

import os
from pathlib import Path

import streamlit as st

from progetto_2.data_handle import DataFrameHandle
from progetto_2.view_sidebar import Sidebar
from progetto_2 import view_panels

# ---------------------------------------------------------------------------
# CONFIG & STYLE
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="E-Commerce Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG_CARD = "#FFFFFF"
NEUTRAL_DARK = "#1B2430"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #F7F8FA;
    }}
    div[data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border: 1px solid #E3E7EC;
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #5B6472;
    }}
    h1, h2, h3 {{
        color: {NEUTRAL_DARK};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Caricamento e pulizia dati in corso (una tantum)...")
def load_handle() -> DataFrameHandle:
    """Carica il CSV e applica la pipeline di cleaning una sola volta per sessione server."""
    root_dir = st.session_state.get("ROOT_DIR")
    res_dir = Path(st.secrets["RES_DIR"])

    res_path = st.session_state["RES_PATH"] = os.path.join(root_dir, res_dir)
    st.session_state["CSV_PATH"] = os.path.join(res_path, st.secrets["PROGETTO_2_CSV"])

    return DataFrameHandle.from_csv(st.session_state["CSV_PATH"])

if "handle" not in st.session_state:
    st.session_state["handle"] = load_handle()
    st.session_state["_initial_filters_applied"] = False

handle: DataFrameHandle = st.session_state["handle"]

# ---------------------------------------------------------------------------
# SIDEBAR — costruita passando l'handle; legge da lì i range/le liste
# ---------------------------------------------------------------------------

# Applica i filtri iniziali solo al primo caricamento (dall'inizio dell'anno corrente)
if not st.session_state["_initial_filters_applied"]:
    sidebar = Sidebar(handle, apply_initial_filters=True)
    st.session_state["_initial_filters_applied"] = True
else:
    sidebar = Sidebar(handle, apply_initial_filters=False)

did_refresh = sidebar.render()

if did_refresh and handle.is_empty:
    st.warning("Nessun dato per i filtri selezionati. Allarga i filtri nella sidebar.")

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

st.title("Dashboard vendite e-commerce")
st.caption(
    "Progetto di portfolio · pipeline completa di data cleaning, analisi statistica "
    "e visualizzazione interattiva con Plotly + Streamlit."
)

if handle.is_empty:
    st.stop()

tab_business, tab_stats, tab_data = st.tabs(
    ["Business Insights", "Analisi statistica", "Dati"]
)

# ---------------------------------------------------------------------------
# PANNELLI — ognuno riceve solo l'handle e legge da lì i dati filtrati
# ---------------------------------------------------------------------------

with tab_business:
    view_panels.render_business_tab(handle)

with tab_stats:
    view_panels.render_stats_tab(handle)

with tab_data:
    view_panels.render_data_tab(handle)

st.divider()
"""
Toolkit di Compliance GDPR / EU AI Act — Streamlit App

Un wizard in 4 passi che guida l'utente a:
  1. Descrivere il sistema AI
  2. Classificarlo secondo il risk-tiering dell'EU AI Act (Regolamento UE 2024/1689)
  3. Verificare i requisiti minimi (GDPR, data governance, trasparenza, human oversight)
  4. Esportare un report di documentazione del rischio in Markdown
"""
import os
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG & STYLE
# ---------------------------------------------------------------------------

from eu_ai_act_toolkit.checklists import filter_applicable, get_all_checklists
from eu_ai_act_toolkit.classification import (
    ANNEX_III_AREAS,
    SCREENING_QUESTIONS,
    TRANSPARENCY_TRIGGERS,
    classify,
)
from eu_ai_act_toolkit.report import generate_markdown_report

st.set_page_config(
    page_title="Toolkit Compliance GDPR / EU AI Act",
    layout="wide",
    initial_sidebar_state="collapsed",
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

if "step" not in st.session_state:
    st.session_state.step = 1
if "checklist_answers" not in st.session_state:
    st.session_state.checklist_answers = {}
if "art5_answers" not in st.session_state:
    st.session_state.art5_answers = {q["key"]: False for q in SCREENING_QUESTIONS}
if "annex3_answers" not in st.session_state:
    st.session_state.annex3_answers = {a["key"]: False for a in ANNEX_III_AREAS}
if "transparency_answers" not in st.session_state:
    st.session_state.transparency_answers = {t["key"]: False for t in TRANSPARENCY_TRIGGERS}


def go_to(step: int):
    st.session_state.step = step


with st.sidebar:
    st.title("Compliance Toolkit")
    st.caption("GDPR · EU AI Act (Regolamento UE 2024/1689)")
    st.divider()

    steps_labels = {
        1: "1️) Descrizione sistema",
        2: "2️) Classificazione del rischio",
        3: "3️) Checklist requisiti",
        4: "4️) Report finale",
    }
    for s, label in steps_labels.items():
        st.button(
            label,
            key=f"nav_{s}",
            use_container_width=True,
            disabled=(s == st.session_state.step),
            on_click=go_to,
            args=(s,),
        )

    st.divider()
    with st.expander("Timeline normativa (aggiornata)"):
        st.markdown(
            """
            - **1 ago 2024** — Entrata in vigore del Regolamento
            - **2 feb 2025** — Divieti Art. 5 e obblighi di AI literacy (Art. 4)
            - **2 ago 2025** — Obblighi per i modelli GPAI (general purpose AI)
            - **2 ago 2026** — Obblighi di trasparenza (Art. 50) ed enforcement UE su GPAI
            - **2 dic 2027** — Obblighi high-risk per sistemi stand-alone (Allegato III),
              a seguito del rinvio disposto dal c.d. *"Digital Omnibus"*
              (Reg. UE 2026/1744, in vigore dal 27 luglio 2026)
            - **2 ago 2028** — Obblighi high-risk per IA incorporata in prodotti già
              regolamentati (Allegato I: dispositivi medici, macchinari, giocattoli, ecc.)

            _Le date dei tier "high-risk" sono state rinviate rispetto al testo
            originale del 2024: verificare sempre la fonte ufficiale
            (EUR-Lex / AI Office) per gli aggiornamenti più recenti._
            """
        )

    st.divider()
    st.caption(
        "ATTENZIONE! Strumento didattico/organizzativo. Non costituisce parere legale. "
        "Progetto dimostrativo per portfolio — AI governance & risk documentation."
    )

st.title("Toolkit di Compliance GDPR / EU AI Act")
st.markdown(
    "Classifica un sistema di intelligenza artificiale secondo il **risk-tiering "
    "dell'EU AI Act** e verifica i requisiti minimi di **trasparenza**, "
    "**data governance** e **human oversight**, generando un report esportabile."
)
st.progress(st.session_state.step / 4)
st.divider()

st.page_link("pages/0_Home.py", label="Torna alla lista progetti")

##############################
# Step 1: Descrizione del sistema
if st.session_state.step == 1:
    st.header("1️) Descrizione del sistema AI")

    st.session_state.system_name = st.text_input(
        "Nome del sistema",
        value=st.session_state.get("system_name", ""),
        placeholder="Es. 'ScreeningCV-AI'",
    )

    st.session_state.provider_or_deployer = st.selectbox(
        "Il tuo ruolo rispetto al sistema",
        ["Fornitore (Provider)", "Utilizzatore (Deployer)", "Entrambi / Non definito"],
        index=["Fornitore (Provider)", "Utilizzatore (Deployer)", "Entrambi / Non definito"].index(
            st.session_state.get("provider_or_deployer", "Fornitore (Provider)")
        ) if st.session_state.get("provider_or_deployer") in
             ["Fornitore (Provider)", "Utilizzatore (Deployer)", "Entrambi / Non definito"] else 0,
        help="Provider = chi sviluppa/immette sul mercato il sistema. "
             "Deployer = chi lo utilizza sotto la propria autorità.",
    )

    st.session_state.system_description = st.text_area(
        "Descrizione funzionale del sistema",
        value=st.session_state.get("system_description", ""),
        height=150,
        placeholder=(
            "Descrivi cosa fa il sistema, quali dati utilizza, chi sono gli utenti "
            "finali e in che contesto viene impiegato (es. 'Sistema di scoring "
            "automatico dei CV in fase di preselezione per posizioni tecniche, "
            "basato su un modello di classificazione addestrato su CV storici...')."
        ),
    )

    st.session_state.author = st.text_input(
        "Compilatore / referente compliance (opzionale)",
        value=st.session_state.get("author", ""),
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Avanti", type="primary"):
            if not st.session_state.system_name.strip():
                st.warning("Inserisci almeno il nome del sistema prima di continuare.")
            else:
                go_to(2)
                st.rerun()

##############################
# Step 2: classificazione del rischio
elif st.session_state.step == 2:
    st.header("2️) Classificazione del rischio (EU AI Act)")
    st.caption(
        "Rispondi alle domande di screening. L'ordine segue l'albero decisionale "
        "del Regolamento: prima i divieti assoluti (Art. 5), poi i casi d'uso "
        "ad alto rischio (Allegato III), infine gli obblighi di trasparenza (Art. 50)."
    )

    with st.expander("🚫 A. Il sistema rientra in una pratica VIETATA? (Art. 5)", expanded=True):
        for q in SCREENING_QUESTIONS:
            st.session_state.art5_answers[q["key"]] = st.checkbox(
                f"{q['text']}  _({q['article']})_",
                value=st.session_state.art5_answers.get(q["key"], False),
                key=f"art5_{q['key']}",
            )

    with st.expander("B. Il sistema rientra in un'area AD ALTO RISCHIO? (Allegato III)"):
        for a in ANNEX_III_AREAS:
            st.session_state.annex3_answers[a["key"]] = st.checkbox(
                a["text"],
                value=st.session_state.annex3_answers.get(a["key"], False),
                key=f"annex3_{a['key']}",
            )
        st.markdown("---")
        st.session_state.is_ancillary = st.checkbox(
            "Il sistema svolge una funzione puramente accessoria/procedurale "
            "rispetto alla decisione umana finale (possibile deroga Art. 6(3))",
            value=st.session_state.get("is_ancillary", False),
            help="Questa deroga NON è applicata automaticamente: va sempre validata "
                 "legalmente e non si applica se il sistema profila persone fisiche.",
        )

    with st.expander("C. Obblighi di trasparenza (Art. 50)"):
        for t in TRANSPARENCY_TRIGGERS:
            st.session_state.transparency_answers[t["key"]] = st.checkbox(
                t["text"],
                value=st.session_state.transparency_answers.get(t["key"], False),
                key=f"transp_{t['key']}",
            )

    st.divider()
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Indietro"):
            go_to(1)
            st.rerun()
    with col2:
        if st.button("Classifica", type="primary"):
            result = classify(
                art5_answers=st.session_state.art5_answers,
                annex3_answers=st.session_state.annex3_answers,
                transparency_answers=st.session_state.transparency_answers,
                is_pure_ancillary_function=st.session_state.get("is_ancillary", False),
            )
            st.session_state.classification_result = result
            go_to(3)
            st.rerun()


##############################
# Step 3: Checklist dei requisiti
elif st.session_state.step == 3:
    st.header("3️) Checklist dei requisiti minimi")

    result = st.session_state.get("classification_result")
    if result is None:
        st.warning("Completa prima lo Step 2 per ottenere una classificazione.")
        if st.button("Vai allo Step 2"):
            go_to(2)
            st.rerun()
        st.stop()

    tier_value = result.tier.value

    tier_colors = {
        "Rischio inaccettabile (VIETATO)": "🔴",
        "Rischio alto (High-Risk)": "🟠",
        "Rischio limitato (obblighi di trasparenza)": "🟡",
        "Rischio minimo/nullo": "🟢",
    }
    st.subheader(f"{tier_colors.get(tier_value, '')} Livello di rischio: {tier_value}")

    with st.expander("Motivazione della classificazione", expanded=True):
        if result.applicable_articles:
            st.markdown("**Riferimenti normativi**: " + ", ".join(result.applicable_articles))
        for r in result.rationale:
            st.info(r)
        if result.triggered_rules:
            st.markdown("**Criteri rilevati:**")
            for rule in result.triggered_rules:
                st.markdown(f"- {rule}")

    if tier_value.startswith("Rischio inaccettabile"):
        st.error(
            "🛑 Il sistema, così descritto, rientra tra le pratiche vietate. "
            "Non è possibile procedere con una checklist di conformità: "
            "il sistema non può essere legalmente utilizzato in questa configurazione."
        )
    else:
        st.caption(
            "Spunta i requisiti già soddisfatti. Vengono mostrati solo quelli "
            "applicabili al livello di rischio individuato. 🔴 = essenziale, 🔵 = consigliato."
        )

        all_checklists = get_all_checklists()
        total_applicable = 0
        total_done = 0

        for title, items in all_checklists.items():
            applicable = filter_applicable(items, tier_value)
            if not applicable:
                continue
            with st.expander(f"**{title}** ({len(applicable)} requisiti applicabili)", expanded=True):
                for item in applicable:
                    total_applicable += 1
                    current = st.session_state.checklist_answers.get(item.id, False)
                    checked = st.checkbox(
                        f"{'🔴' if item.severity == 'essenziale' else '🔵'} {item.text}  _({item.article})_",
                        value=current,
                        key=f"chk_{item.id}",
                    )
                    st.session_state.checklist_answers[item.id] = checked
                    if checked:
                        total_done += 1

        if total_applicable:
            st.metric("Copertura complessiva requisiti", f"{total_done}/{total_applicable}",
                       f"{round(100 * total_done / total_applicable)}%")

    st.divider()
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Indietro"):
            go_to(2)
            st.rerun()
    with col2:
        if st.button("Genera report", type="primary"):
            go_to(4)
            st.rerun()



##############################
# Step 3: Report finale
elif st.session_state.step == 4:
    st.header("4️) Report di documentazione del rischio")

    result = st.session_state.get("classification_result")
    if result is None:
        st.warning("Completa prima gli step precedenti.")
        if st.button("Torna all'inizio"):
            go_to(1)
            st.rerun()
        st.stop()

    report_md = generate_markdown_report(
        system_name=st.session_state.get("system_name", ""),
        system_description=st.session_state.get("system_description", ""),
        provider_or_deployer=st.session_state.get("provider_or_deployer", ""),
        classification=result,
        checklist_answers=st.session_state.checklist_answers,
        author=st.session_state.get("author", ""),
    )

    st.download_button(
        "Scarica report (Markdown)",
        data=report_md,
        file_name=f"risk_documentation_{st.session_state.get('system_name', 'sistema').replace(' ', '_')}.md",
        mime="text/markdown",
        type="primary",
    )

    st.divider()
    st.markdown(report_md)

    st.divider()
    if st.button("Torna alla checklist"):
        go_to(3)
        st.rerun()
    if st.button("Nuova valutazione (reset)"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
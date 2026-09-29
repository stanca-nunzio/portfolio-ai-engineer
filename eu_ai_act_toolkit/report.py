"""
Generazione del report finale di "Documentazione del Rischio" in formato
Markdown, a partire da:
  - metadati del sistema AI
  - risultato della classificazione del rischio
  - stato di completamento delle checklist (per ogni item: True/False/None)

Il report è pensato come punto di partenza per un fascicolo di
accountability interno (non sostituisce la documentazione tecnica formale
richiesta dall'Allegato IV per i sistemi high-risk).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List

from eu_ai_act_toolkit.checklists import ChecklistItem, filter_applicable, get_all_checklists
from eu_ai_act_toolkit.classification import ClassificationResult


def _checklist_section_md(title: str, items: List[ChecklistItem], answers: Dict[str, bool], tier_value: str) -> str:
    applicable = filter_applicable(items, tier_value)
    if not applicable:
        return ""

    lines = [f"### {title}\n"]
    done = 0
    for item in applicable:
        state = answers.get(item.id)
        if state is True:
            box = "[x]"
            done += 1
        elif state is False:
            box = "[ ]"
        else:
            box = "[ ]"
        severity_tag = "🔴 essenziale" if item.severity == "essenziale" else "🔵 consigliato"
        lines.append(f"- {box} **{item.text}** _( {item.article} · {severity_tag} )_")

    coverage = f"{done}/{len(applicable)}" if applicable else "0/0"
    lines.insert(1, f"_Copertura: {coverage} requisiti soddisfatti._\n")
    return "\n".join(lines) + "\n"


def generate_markdown_report(
    system_name: str,
    system_description: str,
    provider_or_deployer: str,
    classification: ClassificationResult,
    checklist_answers: Dict[str, bool],
    author: str = "",
) -> str:
    today = date.today().isoformat()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = []
    md.append(f"# Documentazione del Rischio — {system_name or '(sistema senza nome)'}")
    md.append("")
    md.append(
        "> **Disclaimer**: questo documento è generato da uno strumento di "
        "autovalutazione a scopo dimostrativo/organizzativo. Non costituisce "
        "parere legale né sostituisce la valutazione di conformità formale "
        "richiesta dal Regolamento (UE) 2024/1689 (\"EU AI Act\") o dal GDPR. "
        "Fare riferimento al proprio DPO/consulente legale per la validazione finale."
    )
    md.append("")
    md.append(f"**Data generazione report**: {now}")
    md.append(f"**Autore/compilatore**: {author or '—'}")
    md.append(f"**Ruolo organizzativo**: {provider_or_deployer or '—'}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Descrizione del sistema")
    md.append("")
    md.append(system_description or "_Nessuna descrizione fornita._")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Esito della classificazione del rischio (EU AI Act)")
    md.append("")
    md.append(f"### Livello di rischio: **{classification.tier.value}**")
    md.append("")
    if classification.applicable_articles:
        md.append("**Articoli/riferimenti applicabili**: " + ", ".join(classification.applicable_articles))
        md.append("")
    if classification.triggered_rules:
        md.append("**Criteri che hanno determinato la classificazione**:")
        for rule in classification.triggered_rules:
            md.append(f"- {rule}")
        md.append("")
    if classification.rationale:
        md.append("**Motivazione**:")
        for r in classification.rationale:
            md.append(f"> {r}")
        md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Checklist dei requisiti minimi")
    md.append("")
    md.append(
        "Le checklist seguenti mostrano solo i requisiti **applicabili al livello "
        "di rischio individuato** al punto 2. Le voci contrassegnate 🔴 sono "
        "considerate essenziali; quelle 🔵 sono buone pratiche consigliate."
    )
    md.append("")

    all_checklists = get_all_checklists()
    for title, items in all_checklists.items():
        section = _checklist_section_md(title, items, checklist_answers, classification.tier.value)
        if section:
            md.append(section)

    md.append("---")
    md.append("")
    md.append("## 4. Prossimi passi consigliati")
    md.append("")
    if classification.tier.value.startswith("Rischio inaccettabile"):
        md.append(
            "- 🛑 **Sospendere immediatamente** lo sviluppo/l'uso del sistema in questa configurazione.\n"
            "- Coinvolgere immediatamente il team legale/compliance.\n"
            "- Valutare una riprogettazione che elimini la pratica vietata individuata."
        )
    elif classification.tier.value.startswith("Rischio alto"):
        md.append(
            "- Completare le voci mancanti della checklist, dando priorità a quelle 🔴 essenziali.\n"
            "- Predisporre la documentazione tecnica completa (Allegato IV) e il sistema di gestione del rischio (Art. 9).\n"
            "- Pianificare la valutazione di conformità e, se applicabile, la registrazione nella banca dati UE.\n"
            "- Verificare la necessità di una DPIA (GDPR) e di una FRIA (Art. 27 AI Act).\n"
            "- Coinvolgere DPO/legale per validare la classificazione e il piano di remediation."
        )
    elif classification.tier.value.startswith("Rischio limitato"):
        md.append(
            "- Implementare i meccanismi di trasparenza (disclosure IA, etichettatura contenuti sintetici).\n"
            "- Verificare comunque gli obblighi GDPR generali applicabili.\n"
            "- Rivalutare periodicamente la classificazione in caso di nuove funzionalità."
        )
    else:
        md.append(
            "- Nessun obbligo specifico individuato: si raccomanda comunque "
            "di documentare la valutazione svolta e di ripeterla in caso di modifiche "
            "sostanziali al sistema.\n"
            "- Verificare gli obblighi generali di AI literacy (Art. 4) per il personale coinvolto."
        )

    md.append("")
    md.append("---")
    md.append(f"_Report generato il {today} con il Toolkit di Compliance GDPR / EU AI Act._")

    return "\n".join(md)

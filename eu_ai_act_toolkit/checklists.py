"""
Checklist strutturate di requisiti minimi, organizzate per area:

  - GDPR (Regolamento (UE) 2016/679) — principi generali applicabili quando il
    sistema AI tratta dati personali
  - AI Act · Data Governance (Art. 10) — qualità e governo dei dati di addestramento
  - AI Act · Trasparenza (Art. 13 per i sistemi high-risk, Art. 50 per i sistemi
    a rischio limitato)
  - AI Act · Human Oversight (Art. 14) — sorveglianza umana sui sistemi high-risk
  - AI Act · Gestione del rischio & documentazione tecnica (Art. 9, Art. 11,
    Allegato IV)

Ogni voce ha:
  - id            : identificativo univoco (usato per salvare lo stato nella sessione)
  - text          : il requisito in linguaggio naturale
  - article       : riferimento normativo indicativo
  - applies_to    : tier minimi a cui il requisito si applica (per filtrare la
                     checklist in base al risultato della classificazione)
  - severity      : "essenziale" | "consigliato" — utile per dare priorità
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ChecklistItem:
    id: str
    text: str
    article: str
    applies_to: List[str]  # valori compatibili con RiskTier.value o "ALL"
    severity: str  # "essenziale" | "consigliato"


GDPR_CHECKLIST: List[ChecklistItem] = [
    ChecklistItem(
        id="gdpr_legal_basis",
        text="È stata individuata e documentata una base giuridica (art. 6 GDPR, ed "
             "eventualmente art. 9 se dati particolari) per ogni trattamento di dati "
             "personali effettuato dal sistema.",
        article="Art. 6, 9 GDPR",
        applies_to=["ALL"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="gdpr_dpia",
        text="È stata condotta (o pianificata) una Valutazione d'Impatto sulla "
             "Protezione dei Dati (DPIA) se il trattamento presenta un rischio elevato "
             "per i diritti e le libertà delle persone.",
        article="Art. 35 GDPR",
        applies_to=["Rischio alto (High-Risk)", "Rischio limitato (obblighi di trasparenza)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="gdpr_minimization",
        text="I dati raccolti/utilizzati per addestramento o inferenza sono limitati "
             "a quanto necessario allo scopo (minimizzazione dei dati).",
        article="Art. 5(1)(c) GDPR",
        applies_to=["ALL"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="gdpr_retention",
        text="Sono definiti e documentati i tempi di conservazione dei dati "
             "personali trattati dal sistema, con criteri di cancellazione.",
        article="Art. 5(1)(e) GDPR",
        applies_to=["ALL"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="gdpr_data_subject_rights",
        text="Sono predisposte procedure per gestire le richieste di esercizio dei "
             "diritti degli interessati (accesso, rettifica, cancellazione, opposizione, "
             "portabilità) anche quando i dati sono trattati da un sistema AI.",
        article="Artt. 15-22 GDPR",
        applies_to=["ALL"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="gdpr_automated_decisions",
        text="Se il sistema produce decisioni basate unicamente su trattamento "
             "automatizzato con effetti giuridici o significativi, sono garantiti "
             "il diritto a intervento umano, a esprimere la propria opinione e a "
             "contestare la decisione (art. 22 GDPR).",
        article="Art. 22 GDPR",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="gdpr_dpo",
        text="È stato coinvolto/consultato il Responsabile della Protezione dei Dati "
             "(DPO), ove nominato, nella progettazione del sistema.",
        article="Artt. 37-39 GDPR",
        applies_to=["ALL"],
        severity="consigliato",
    ),
    ChecklistItem(
        id="gdpr_international_transfer",
        text="Se i dati vengono trasferiti fuori dallo SEE (es. provider cloud extra-UE), "
             "è presente una base giuridica per il trasferimento (decisione di adeguatezza, "
             "SCC, BCR).",
        article="Artt. 44-49 GDPR",
        applies_to=["ALL"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="gdpr_security_measures",
        text="Sono implementate misure tecniche e organizzative adeguate a garantire "
             "la sicurezza dei dati trattati (cifratura, controllo accessi, logging).",
        article="Art. 32 GDPR",
        applies_to=["ALL"],
        severity="essenziale",
    ),
]

DATA_GOVERNANCE_CHECKLIST: List[ChecklistItem] = [
    ChecklistItem(
        id="dg_data_quality",
        text="I set di dati di addestramento, validazione e test sono pertinenti, "
             "rappresentativi e il più possibile privi di errori, per quanto possibile "
             "rispetto alla finalità prevista.",
        article="Art. 10(3) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="dg_bias_examination",
        text="I dataset sono stati esaminati per individuare possibili bias che "
             "potrebbero condurre a discriminazioni vietate, con misure di mitigazione "
             "adottate ove necessario.",
        article="Art. 10(2)(f)-(g) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="dg_provenance",
        text="È documentata la provenienza dei dati (fonte, modalità di raccolta, "
             "eventuale consenso o altra base giuridica) per i dataset di training.",
        article="Art. 10(2)(b)-(c) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="dg_special_categories",
        text="Se vengono trattate categorie particolari di dati per correggere bias, "
             "sono in essere misure di sicurezza aggiuntive e limitazioni d'uso "
             "specifiche previste dall'Art. 10(5).",
        article="Art. 10(5) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="consigliato",
    ),
    ChecklistItem(
        id="dg_versioning",
        text="Sono tracciate le versioni dei dataset e dei modelli, con possibilità "
             "di ricostruire quale versione di dati/modello ha prodotto un dato output.",
        article="Art. 12 AI Act (record-keeping)",
        applies_to=["Rischio alto (High-Risk)"],
        severity="consigliato",
    ),
]

TRANSPARENCY_CHECKLIST: List[ChecklistItem] = [
    ChecklistItem(
        id="tr_ai_disclosure",
        text="Le persone fisiche sono informate chiaramente che stanno interagendo "
             "con un sistema di intelligenza artificiale (es. chatbot) prima o al "
             "momento della prima interazione.",
        article="Art. 50(1) AI Act",
        applies_to=["Rischio limitato (obblighi di trasparenza)", "Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="tr_synthetic_content_label",
        text="I contenuti sintetici (immagini, audio, video, testo) generati o "
             "manipolati dal sistema sono etichettati in modo leggibile da "
             "una macchina e riconoscibile come artificiali/manipolati.",
        article="Art. 50(2)(4) AI Act",
        applies_to=["Rischio limitato (obblighi di trasparenza)", "Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="tr_instructions_for_use",
        text="Sono fornite istruzioni per l'uso complete: finalità prevista, livello "
             "di accuratezza, limitazioni note, condizioni di funzionamento previste.",
        article="Art. 13(3) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="tr_deployer_info",
        text="Il fornitore mette a disposizione del deployer (utilizzatore a valle) "
             "informazioni sufficienti a comprendere il funzionamento del sistema e "
             "ad interpretarne correttamente l'output.",
        article="Art. 13(1)-(2) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="tr_capabilities_limits",
        text="Sono documentate esplicitamente le capacità e i limiti di prestazione "
             "del sistema, incluse le circostanze che possono comportare rischi per "
             "la salute, la sicurezza o i diritti fondamentali.",
        article="Art. 13(3)(b) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
]

HUMAN_OVERSIGHT_CHECKLIST: List[ChecklistItem] = [
    ChecklistItem(
        id="ho_named_overseers",
        text="Sono individuate le persone fisiche incaricate della sorveglianza "
             "umana, con formazione adeguata e autorità/tempo/risorse necessari per "
             "esercitarla effettivamente.",
        article="Art. 14(4)(a)-(b) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="ho_understand_capabilities",
        text="Chi sorveglia il sistema comprende adeguatamente le sue capacità e "
             "i suoi limiti, ed è in grado di monitorare correttamente il suo "
             "funzionamento (compreso il rilevamento di anomalie/malfunzionamenti).",
        article="Art. 14(4)(a) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="ho_override_capability",
        text="Gli operatori umani hanno la possibilità concreta di non tenere conto, "
             "ignorare o annullare l'output del sistema, o di interromperne il "
             "funzionamento (kill switch/stop button) quando necessario.",
        article="Art. 14(4)(d)-(e) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="ho_automation_bias",
        text="Sono adottate misure per mitigare il rischio di 'automation bias' "
             "(eccessivo affidamento acritico sull'output del sistema da parte "
             "degli operatori umani).",
        article="Art. 14(4)(c) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="ho_interpretation_tools",
        text="Sono disponibili strumenti di interpretazione dell'output (es. "
             "spiegazioni, punteggi di confidenza) che supportino una revisione "
             "umana informata, ove pertinente.",
        article="Art. 14(4)(a) AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="consigliato",
    ),
]

RISK_MGMT_DOCS_CHECKLIST: List[ChecklistItem] = [
    ChecklistItem(
        id="rm_risk_management_system",
        text="È istituito un sistema di gestione del rischio, mantenuto e "
             "aggiornato lungo l'intero ciclo di vita del sistema (identificazione, "
             "stima, valutazione e mitigazione dei rischi noti e prevedibili).",
        article="Art. 9 AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="rm_technical_documentation",
        text="È redatta la documentazione tecnica richiesta (Allegato IV): "
             "descrizione generale, elementi di progettazione e sviluppo, "
             "informazioni su monitoraggio/funzionamento/controllo.",
        article="Art. 11, Allegato IV AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="rm_logging",
        text="Il sistema è in grado di registrare automaticamente log (event logs) "
             "durante il suo funzionamento, per un livello di tracciabilità adeguato "
             "al rischio.",
        article="Art. 12 AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="rm_accuracy_robustness",
        text="Sono definiti e testati livelli adeguati di accuratezza, robustezza "
             "e cybersecurity, coerenti con la finalità prevista del sistema.",
        article="Art. 15 AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="rm_conformity_assessment",
        text="È stata pianificata/completata la procedura di valutazione della "
             "conformità applicabile, con eventuale coinvolgimento di un organismo "
             "notificato dove richiesto.",
        article="Art. 43 AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
    ChecklistItem(
        id="rm_eu_database",
        text="È pianificata la registrazione del sistema nella banca dati UE per i "
             "sistemi di IA ad alto rischio, ove applicabile (fornitori/deployer "
             "enti pubblici).",
        article="Art. 49, 71 AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="consigliato",
    ),
    ChecklistItem(
        id="rm_fria",
        text="Se il deployer è un ente pubblico, un soggetto privato che fornisce "
             "servizi pubblici essenziali, o il sistema è usato per credit scoring/"
             "assicurazioni, è stata condotta una Valutazione d'Impatto sui Diritti "
             "Fondamentali (FRIA).",
        article="Art. 27 AI Act",
        applies_to=["Rischio alto (High-Risk)"],
        severity="essenziale",
    ),
]


def get_all_checklists() -> dict:
    return {
        "GDPR": GDPR_CHECKLIST,
        "Data Governance (AI Act)": DATA_GOVERNANCE_CHECKLIST,
        "Trasparenza (AI Act)": TRANSPARENCY_CHECKLIST,
        "Human Oversight (AI Act)": HUMAN_OVERSIGHT_CHECKLIST,
        "Gestione del Rischio & Documentazione (AI Act)": RISK_MGMT_DOCS_CHECKLIST,
    }


def filter_applicable(items: List[ChecklistItem], tier_value: str) -> List[ChecklistItem]:
    """Ritorna solo le voci applicabili al tier di rischio indicato (o 'ALL')."""
    return [it for it in items if "ALL" in it.applies_to or tier_value in it.applies_to]

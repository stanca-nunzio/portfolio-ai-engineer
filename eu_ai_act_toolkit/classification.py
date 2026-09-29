"""
Motore di classificazione del rischio ai sensi del Regolamento (UE) 2024/1689
("EU AI Act").

ATTENZIONE — DISCLAIMER
Questo modulo NON costituisce parere legale. È uno strumento didattico/di
supporto pensato per orientare una prima autovalutazione interna (screening).
Per una classificazione vincolante rivolgersi a un legale specializzato in
diritto digitale/AI governance o al proprio DPO.

Il modulo implementa una versione semplificata ma fedele nella struttura
logica dell'albero decisionale previsto dal Regolamento:

    1) Pratica vietata (Art. 5)?              -> INACCETTABILE
    2) Rientra in un caso d'uso Allegato III   -> HIGH RISK (salvo eccezioni
       o in interagisce con persone fisiche       art. 6(3): funzione puramente
       come sistemi di IA per finalità              accessoria/procedurale)
       generali con rischio sistemico
       (GPAI) o componente di sicurezza
       di un prodotto già regolamentato
       (Allegato I)?
    3) Interagisce con persone fisiche,
       genera contenuti sintetici, o fa
       riconoscimento emozioni/biometrico
       non vietato?                          -> RISCHIO LIMITATO (trasparenza, Art. 50)
    4) Altrimenti                             -> RISCHIO MINIMO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class RiskTier(str, Enum):
    UNACCEPTABLE = "Rischio inaccettabile (VIETATO)"
    HIGH = "Rischio alto (High-Risk)"
    LIMITED = "Rischio limitato (obblighi di trasparenza)"
    MINIMAL = "Rischio minimo/nullo"


@dataclass
class ClassificationAnswer:
    key: str
    label: str
    value: bool


@dataclass
class ClassificationResult:
    tier: RiskTier
    triggered_rules: List[str] = field(default_factory=list)
    applicable_articles: List[str] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)


# DOMANDE DEL QUESTIONARIO DI SCREENING
# Ogni domanda è una euristica di alto livello derivata da Art. 5 e Allegato III.
# In un tool di produzione andrebbero mappate 1:1 sulle lettere dell'Allegato III
# e sulle 8 pratiche vietate elencate all'Art. 5; qui sono aggregate per usabilità.

SCREENING_QUESTIONS = [
    {
        "key": "social_scoring",
        "text": "Il sistema valuta o classifica persone fisiche in base al comportamento "
                "sociale o a caratteristiche personali, con effetti pregiudizievoli o "
                "sproporzionati in contesti non collegati (social scoring)?",
        "article": "Art. 5(1)(c)",
    },
    {
        "key": "subliminal_manipulation",
        "text": "Il sistema utilizza tecniche subliminali, manipolative o ingannevoli "
                "che alterano materialmente il comportamento di una persona causandole "
                "un danno significativo?",
        "article": "Art. 5(1)(a)",
    },
    {
        "key": "vulnerability_exploitation",
        "text": "Il sistema sfrutta vulnerabilità legate a età, disabilità o condizione "
                "socio-economica per distorcere il comportamento causando un danno?",
        "article": "Art. 5(1)(b)",
    },
    {
        "key": "biometric_categorization",
        "text": "Il sistema categorizza persone fisiche in base a dati biometrici per "
                "inferire origine razziale/etnica, opinioni politiche, orientamento "
                "sessuale o altre categorie protette?",
        "article": "Art. 5(1)(g)",
    },
    {
        "key": "realtime_biometric_public",
        "text": "Il sistema effettua identificazione biometrica remota 'in tempo reale' "
                "in spazi accessibili al pubblico a fini di attività di contrasto (law "
                "enforcement)?",
        "article": "Art. 5(1)(h)",
    },
    {
        "key": "emotion_recognition_workplace_edu",
        "text": "Il sistema effettua riconoscimento delle emozioni sul luogo di lavoro o "
                "in ambito scolastico (salvo eccezioni per motivi medici/sicurezza)?",
        "article": "Art. 5(1)(f)",
    },
    {
        "key": "predictive_policing",
        "text": "Il sistema effettua valutazioni predittive del rischio che una persona "
                "fisica commetta un reato basandosi unicamente sulla profilazione o su "
                "tratti di personalità?",
        "article": "Art. 5(1)(d)",
    },
    {
        "key": "facial_scraping",
        "text": "Il sistema crea o amplia database di riconoscimento facciale tramite "
                "scraping non mirato da internet o da telecamere CCTV?",
        "article": "Art. 5(1)(e)",
    },
]

# Casi d'uso Allegato III (versione semplificata, aggregata per area)
ANNEX_III_AREAS = [
    {
        "key": "biometrics_annex3",
        "text": "Identificazione biometrica (non vietata) o categorizzazione biometrica "
                "usata per l'accesso a servizi essenziali?",
    },
    {
        "key": "critical_infrastructure",
        "text": "Il sistema gestisce/protegge infrastrutture critiche (energia, acqua, "
                "trasporti, reti digitali) come componente di sicurezza?",
    },
    {
        "key": "education",
        "text": "Il sistema determina l'accesso, l'ammissione o la valutazione di studenti "
                "in percorsi educativi/formativi (es. correzione automatica di esami)?",
    },
    {
        "key": "employment",
        "text": "Il sistema è usato per selezione/reclutamento, promozione, licenziamento "
                "o monitoraggio delle performance dei lavoratori?",
    },
    {
        "key": "essential_services",
        "text": "Il sistema determina l'accesso a servizi essenziali pubblici o privati "
                "(es. credit scoring, prestazioni sociali, assicurazioni sanitarie/vita)?",
    },
    {
        "key": "law_enforcement",
        "text": "Il sistema è usato dalle forze dell'ordine per valutare rischi, "
                "affidabilità di prove, o profilazione nell'ambito di indagini?",
    },
    {
        "key": "migration_justice",
        "text": "Il sistema è usato in ambito migrazione/asilo/controllo di frontiera, "
                "oppure per assistere autorità giudiziarie nell'interpretazione della legge?",
    },
    {
        "key": "democratic_processes",
        "text": "Il sistema è destinato a influenzare l'esito di elezioni o referendum, "
                "o il comportamento di voto delle persone fisiche?",
    },
]

TRANSPARENCY_TRIGGERS = [
    {
        "key": "interacts_with_humans",
        "text": "Il sistema interagisce direttamente con persone fisiche (es. chatbot, "
                "assistente virtuale)?",
    },
    {
        "key": "generates_synthetic_content",
        "text": "Il sistema genera o manipola contenuti (immagini, audio, video, testo) "
                "che potrebbero apparire come autentici (es. deepfake, contenuti generativi)?",
    },
    {
        "key": "emotion_or_biometric_categorization_other",
        "text": "Il sistema effettua riconoscimento delle emozioni o categorizzazione "
                "biometrica in contesti diversi da quelli già vietati/high-risk sopra?",
    },
]


def classify(
    art5_answers: dict,
    annex3_answers: dict,
    transparency_answers: dict,
    is_pure_ancillary_function: bool = False,
) -> ClassificationResult:
    """
    Applica l'albero decisionale semplificato dell'EU AI Act.

    Parameters
    ----------
    art5_answers: dict[str, bool]
        risposte alle domande su pratiche vietate (chiavi = SCREENING_QUESTIONS[i]['key'])
    annex3_answers: dict[str, bool]
        risposte sulle aree Allegato III (chiavi = ANNEX_III_AREAS[i]['key'])
    transparency_answers: dict[str, bool]
        risposte sui trigger di trasparenza (chiavi = TRANSPARENCY_TRIGGERS[i]['key'])
    is_pure_ancillary_function: bool
        se True, l'utente dichiara che il sistema svolge una funzione puramente
        accessoria/procedurale rispetto alla decisione umana (art. 6(3) — deroga
        che può derubricare un caso Allegato III da high-risk, salvo che effettui
        comunque profilazione di persone fisiche).

    Returns
    -------
    ClassificationResult
    """
    triggered_rules: List[str] = []
    applicable_articles: List[str] = []
    rationale: List[str] = []

    # 1) Pratiche vietate
    prohibited_hits = [k for k, v in art5_answers.items() if v]
    if prohibited_hits:
        for q in SCREENING_QUESTIONS:
            if q["key"] in prohibited_hits:
                triggered_rules.append(q["text"])
                applicable_articles.append(q["article"])
        rationale.append(
            "Almeno una pratica rientra tra quelle vietate ai sensi dell'Art. 5. "
            "Il sistema NON può essere immesso sul mercato, messo in servizio o "
            "utilizzato nell'UE in questa configurazione."
        )
        return ClassificationResult(
            tier=RiskTier.UNACCEPTABLE,
            triggered_rules=triggered_rules,
            applicable_articles=sorted(set(applicable_articles)),
            rationale=rationale,
        )

    # 2) Allegato III (high-risk)
    annex3_hits = [k for k, v in annex3_answers.items() if v]
    if annex3_hits:
        for area in ANNEX_III_AREAS:
            if area["key"] in annex3_hits:
                triggered_rules.append(area["text"])

        if is_pure_ancillary_function:
            rationale.append(
                "Il sistema rientra in un'area dell'Allegato III, ma è stata dichiarata "
                "una funzione puramente accessoria/procedurale rispetto alla decisione "
                "umana (deroga Art. 6(3)). ATTENZIONE: questa deroga NON si applica se il "
                "sistema effettua comunque profilazione di persone fisiche, e la valutazione "
                "va documentata prima della messa in servizio. Si raccomanda comunque "
                "una verifica legale puntuale prima di derubricare il sistema."
            )
            # Non deroghiamo automaticamente: mostriamo un tier "condizionale" restando
            # prudenti, dato che l'auto-dichiarazione dell'utente non è verifica legale.
            triggered_rules.append(
                "Deroga Art. 6(3) dichiarata dall'utente: da validare legalmente, "
                "non applicata automaticamente da questo tool."
            )

        applicable_articles.extend(["Art. 6", "Art. 8-15 (requisiti high-risk)", "Allegato III"])
        rationale.append(
            "Il sistema rientra in almeno un caso d'uso dell'Allegato III e non risulta "
            "vietato: si applicano i requisiti pieni per i sistemi ad alto rischio "
            "(gestione del rischio, data governance, documentazione tecnica, "
            "trasparenza verso il deployer, human oversight, accuratezza/robustezza/"
            "cybersecurity, registrazione nel database UE)."
        )
        return ClassificationResult(
            tier=RiskTier.HIGH,
            triggered_rules=triggered_rules,
            applicable_articles=sorted(set(applicable_articles)),
            rationale=rationale,
        )

    # 3) Rischio limitato (trasparenza)
    transparency_hits = [k for k, v in transparency_answers.items() if v]
    if transparency_hits:
        for t in TRANSPARENCY_TRIGGERS:
            if t["key"] in transparency_hits:
                triggered_rules.append(t["text"])
        applicable_articles.append("Art. 50 (obblighi di trasparenza)")
        rationale.append(
            "Il sistema non è vietato né high-risk, ma interagisce con persone fisiche "
            "e/o genera contenuti sintetici: si applicano gli obblighi di trasparenza "
            "(informare l'utente che sta interagendo con un'IA, etichettare i contenuti "
            "sintetici/deepfake)."
        )
        return ClassificationResult(
            tier=RiskTier.LIMITED,
            triggered_rules=triggered_rules,
            applicable_articles=sorted(set(applicable_articles)),
            rationale=rationale,
        )

    # 4) Rischio minimo
    rationale.append(
        "In base alle risposte fornite, il sistema non rientra in nessuna delle "
        "categorie vietate, ad alto rischio o soggette a obblighi di trasparenza "
        "specifici. Nessun obbligo ai sensi dell'AI Act, fermi restando gli "
        "eventuali obblighi generali di AI literacy (Art. 4) e le normative "
        "settoriali/GDPR applicabili."
    )
    return ClassificationResult(
        tier=RiskTier.MINIMAL,
        triggered_rules=[],
        applicable_articles=["Art. 4 (AI literacy) — obbligo generale"],
        rationale=rationale,
    )

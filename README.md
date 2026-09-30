# Portfolio RAG - Retrieval Augmented Generation

Progetto dimostrativo che mostra come costruire un sistema di domande e
risposte su documenti (PDF/Markdown) usando ricerca semantica e un LLM.
Fa parte di un portfolio Streamlit multi-pagina.

## Dati e fonti

Al primo avvio, se la cartella `data/` è vuota, l'app scarica automaticamente
alcuni documenti di esempio da [data.europa.eu](https://data.europa.eu) (il
portale ufficiale dei dati aperti dell'Unione Europea), per popolare l'indice
RAG con contenuti reali su cui fare domande. L'elenco dei documenti scaricati
è configurabile in `documents.json`.

Documenti attualmente inclusi:

- [Data Spaces Panel Report](https://data.europa.eu/sites/default/files/report/Data_Spaces_Panel_Report_EN.pdf)
- [Re-using Open Data](https://data.europa.eu/sites/default/files/re-using_open_data.pdf)
- [Citizen-generated data on data.europa.eu](https://data.europa.eu/sites/default/files/report/data.europa.eu_Report_Citizen-generateddataondata_europa_eu.pdf)

© European Union, [data.europa.eu](https://data.europa.eu). Documenti
riutilizzati ai sensi della [Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), la licenza di
default adottata da data.europa.eu salvo diversa indicazione sulla singola
pagina di pubblicazione. Nell'app, la fonte di ogni documento scaricato
automaticamente è indicata anche nella tab "Files", sotto l'anteprima del
file.

Questi documenti sono usati esclusivamente a scopo dimostrativo, per mostrare
il funzionamento della pipeline RAG (chunking, embedding, retrieval,
generazione) su contenuti testuali reali; non rappresentano un'analisi o
un'elaborazione ufficiale dei report citati.

## Cos'è un RAG, in breve

Un LLM può rispondere solo con quello che sa dal training, quindi non
conosce i tuoi documenti privati. Il RAG risolve il problema in due fasi:

1. **Indicizzazione**: i documenti vengono spezzati in pezzi (chunk),
   trasformati in vettori numerici (embedding) e salvati in un database
   vettoriale.
2. **Query**: quando l'utente fa una domanda, la si trasforma anch'essa
   in un vettore, si cercano i chunk più simili nel database, e li si
   passa come contesto all'LLM insieme alla domanda originale. L'LLM
   genera la risposta basandosi su quel contesto, invece che sulla sua
   conoscenza generica.

Il vantaggio: risposte ancorate a fonti verificabili, aggiornabili senza
dover riaddestrare nessun modello.

## Architettura

```
data/                      documenti sorgente (PDF, Markdown)
        |
        v
rag_system/ingest.py        estrazione testo + chunking a finestra scorrevole
        |
        v
rag_system/vectorstore.py   embedding (sentence-transformers) + Qdrant embedded
        |
        v
pages/1_RAG.py         UI Streamlit: domanda utente -> ricerca -> risposta
        |
        v
rag_system/llm.py           chiamata a Google Gemini con contesto recuperato
```

## Come funziona il flusso, passo per passo

1. All'avvio della pagina RAG, `setup_index()` (decorata con
   `st.cache_resource`, quindi eseguita una sola volta per sessione del
   server) legge tutti i file da `data/` con `load_documents`.
2. Ogni documento viene spezzato in chunk da `chunk_text` (finestra
   scorrevole su caratteri, con overlap configurabile via secrets).
3. `index_chunks` calcola l'embedding di ogni chunk e lo salva su Qdrant,
   insieme al testo originale come payload (evita di riaprire i file al
   momento della risposta).
4. Quando l'utente scrive una domanda e clicca "Chiedi", `search` cerca
   i chunk più simili per similarità coseno.
5. `generate_answer` costruisce un prompt con i chunk trovati come
   contesto e lo invia a Gemini, che genera la risposta finale citando
   le fonti quando possibile.
6. L'interfaccia mostra la risposta, le fonti usate e, in un expander,
   i chunk grezzi recuperati (utile per debug e per capire cosa "vede"
   il modello).

## Configurazione

Le variabili richieste vanno messe in `.streamlit/secrets.toml`:

```toml
DOCS_DIR = "data"
CHUNK_SIZE = "800"
CHUNK_OVERLAP = "100"
QDRANT_PATH = "qdrant_storage"
GOOGLE_API_KEY = "..."
LLM_MODEL_ASK = "gemini-2.0-flash"
```

## Avvio in locale

```bash
pip install -r requirements.txt
streamlit run app.py
```
# Portfolio LangGraph - Assistente di scrittura con scaletta interattiva

Progetto dimostrativo che mostra come costruire un agente con **LangGraph** e
**Google Gemini** in cui l'utente resta nel ciclo (human-in-the-loop): il
modello propone una scaletta, l'utente la modifica a piacere e solo dopo la sua
approvazione viene scritto il testo completo. Ogni passaggio è protetto da
controlli di sicurezza. Fa parte di un portfolio Streamlit multi-pagina.

> **Nota**: i guardrail riducono i rischi ma nessuno è perfetto da solo. Il
> progetto è dimostrativo e i testi generati da un modello di IA vanno sempre
> verificati prima di essere usati.

## Cosa mostra il progetto

1. **Grafo con pause**: il flusso si ferma in attesa dell'utente (`interrupt`)
   e riparte con la sua scelta, grazie a un checkpointer che conserva lo stato
2. **Più modelli con ruoli diversi**: Guardia, Outliner e Writer hanno system
   prompt e temperature dedicati
3. **Guardrail a più livelli**: limiti di input, moderazione con output
   strutturato, regole nei prompt, safety settings nativi di Gemini
4. **UI in streaming**: la pagina Streamlit mostra in tempo reale i passaggi
   del grafo, compresi i controlli di sicurezza

## I ruoli

| Ruolo | Temperatura | Compito |
|---|---|---|
| Guardia | 0 | Moderatore: valuta la richiesta prima di ogni elaborazione e il testo finale prima di mostrarlo |
| Outliner | 0.4 | Crea la scaletta (4-8 sezioni) e la aggiorna in base alle richieste dell'utente |
| Writer | 0.6 | Scrive il testo completo seguendo fedelmente la scaletta approvata |

## Architettura

```
langgraph_agents/graph.py    grafo LangGraph: nodi, routing, guardrail,
        |                    funzioni start_session / resume_session
        v
pages/4_LangGraph.py         UI Streamlit: prompt, modifica scaletta, testo finale
```

Il grafo, con due pause per l'utente:

```
START -> guard_input -(ok)-> outliner -> review   (PAUSA: l'utente decide)
             |                              |
        (bloccato)                  "refine" -> guard_input -> outliner -> review ...
             v                              |
        END / review                "write"  -> writer -> guard_output -> END
```

## Guardrail

1. **Limiti**: massimo 2000 caratteri per prompt e istruzioni, massimo 6
   modifiche alla scaletta per sessione
2. **Guardia sull'input**: un LLM con output strutturato valuta prompt e
   modifiche. In caso di errore del controllo la richiesta viene bloccata
   (*fail-closed*)
3. **Regole nel system prompt** di Outliner e Writer: se la richiesta le viola,
   il modello risponde con il marcatore `[RIFIUTATO]` e il grafo lo intercetta
4. **Safety settings nativi di Gemini** (se disponibili nella versione
   installata di `langchain-google-genai`)
5. **Guardia sull'output**: il testo finale viene controllato prima di essere
   mostrato

Sono bloccati contenuti violenti, degradanti o d'odio, istruzioni per costruire
oggetti pericolosi, contenuti sessuali espliciti, autolesionismo, attività
illegali, dati personali di privati e tentativi di aggirare le regole. Temi
sensibili (storia, cronaca, scienza, prevenzione) restano ammessi a livello
informativo, senza dettagli operativi replicabili.

## Come funziona il flusso, passo per passo

1. **Prompt**: l'utente descrive cosa vuole scrivere. `guard_input` controlla
   la richiesta; se è accettata, `outliner` produce la scaletta (v1)
2. **Pausa** (`review`): il grafo si ferma con `interrupt` e la UI mostra la
   scaletta a destra
3. **Modifiche**: l'utente scrive un'istruzione ("riassumi", "sposta la sezione
   3 all'inizio"). L'istruzione torna a `guard_input` (valutata insieme
   all'argomento originale) e poi a `outliner`, che restituisce l'intera
   scaletta aggiornata. Si può ripetere fino a 6 volte
4. **Approvazione**: con «Produci il testo» il `writer` scrive il testo in
   Markdown seguendo la scaletta
5. **Controllo finale**: `guard_output` verifica il testo; se è sicuro viene
   mostrato e scaricabile in `.md`, altrimenti viene bloccato e resta visibile
   la scaletta

## Configurazione

Le variabili vanno messe in `.streamlit/secrets.toml` (in alternativa come
variabili d'ambiente):

```toml
GOOGLE_API_KEY = "..."
GEMINI_MODELS = ["gemini-2.0-flash"]
```

`GEMINI_MODELS` è l'elenco dei modelli selezionabili nella sidebar; come
variabile d'ambiente si scrive separato da virgole. Se manca, viene usato
`gemini-2.0-flash`.

## Limiti noti

- Lo stato è in memoria: al riavvio del server le sessioni si perdono. Per la
  persistenza si può sostituire `MemorySaver` con un checkpointer SQLite o
  Postgres
- La Guardia è un LLM e può dare falsi positivi o falsi negativi



# Portfolio Compliance - Toolkit GDPR / EU AI Act

Progetto dimostrativo che guida l'utente nella classificazione di un sistema
di intelligenza artificiale secondo il risk-tiering dell'EU AI Act
(Regolamento UE 2024/1689) e nella verifica dei requisiti minimi di
conformità, fino alla produzione di un report di documentazione del rischio.
Fa parte di un portfolio Streamlit multi-pagina.

> **Nota**: strumento didattico/organizzativo, realizzato a scopo
> dimostrativo (AI governance & risk documentation). Non costituisce parere
> legale e non sostituisce una valutazione di conformità condotta da
> professionisti.

## Cosa mostra il progetto

Un wizard in 4 passi che replica, in forma semplificata, il ragionamento di
un referente compliance davanti a un nuovo sistema AI:

1. **Descrizione**: nome del sistema, ruolo dell'utente (Provider, Deployer o
   entrambi), descrizione funzionale, compilatore
2. **Classificazione**: domande di screening che seguono l'albero decisionale
   del Regolamento, dal livello più grave al meno grave
3. **Checklist**: requisiti minimi (GDPR, data governance, trasparenza, human
   oversight) filtrati in base al livello di rischio individuato
4. **Report**: documento di documentazione del rischio, visualizzabile
   nell'app e scaricabile in Markdown

## Livelli di rischio

L'ordine delle domande di screening rispecchia la gerarchia del Regolamento:

| Livello | Riferimento | Esito |
|---|---|---|
| Rischio inaccettabile (VIETATO) | Art. 5 | Pratica vietata: la checklist non viene proposta |
| Rischio alto (High-Risk) | Art. 6, Allegato III | Checklist completa dei requisiti |
| Rischio limitato | Art. 50 | Obblighi di trasparenza |
| Rischio minimo/nullo | - | Nessun obbligo specifico dal Regolamento |

La deroga per le funzioni puramente accessorie/procedurali (Art. 6(3)) è
offerta come opzione ma **non viene mai applicata automaticamente**: va
sempre validata legalmente e non si applica ai sistemi che profilano persone
fisiche.

## Architettura

```
eu_ai_act_toolkit/classification.py   domande di screening (Art. 5, Allegato III,
        |                             Art. 50) + funzione classify()
        v
eu_ai_act_toolkit/checklists.py       checklist dei requisiti, filtrabili per
        |                             livello di rischio (filter_applicable)
        v
eu_ai_act_toolkit/report.py           generate_markdown_report
        |
        v
pages/3_AI_Act_Toolkit.py             entry point della pagina Streamlit
                                      (wizard a 4 step)
```

## Come funziona il flusso, passo per passo

1. **Step 1**: l'utente inserisce nome, ruolo e descrizione del sistema. Il
   nome è obbligatorio per proseguire.
2. **Step 2**: tre gruppi di domande, con checkbox: pratiche vietate
   (Art. 5), aree ad alto rischio (Allegato III, più la deroga Art. 6(3)) e
   trigger di trasparenza (Art. 50). Al click su "Classifica", `classify`
   restituisce livello, articoli applicabili, motivazione e criteri
   rilevati.
3. **Step 3**: viene mostrato il livello di rischio con la sua motivazione.
   Se il sistema è vietato la checklist è bloccata; altrimenti compaiono i
   soli requisiti applicabili al livello, marcati come essenziali (🔴) o
   consigliati (🔵). Una metrica riassume la copertura complessiva
   (requisiti soddisfatti / applicabili).
4. **Step 4**: `generate_markdown_report` compone il report a partire da
   descrizione, classificazione e risposte alla checklist; l'utente può
   leggerlo nella pagina e scaricarlo come file `.md`. Il pulsante "Nuova
   valutazione" azzera lo stato.

Lo stato del wizard (step corrente, risposte, risultato della
classificazione) è tenuto in `st.session_state`. La navigazione dalla
sidebar usa callback `on_click`, così il pulsante dello step corrente risulta
sempre disabilitato correttamente.

## Configurazione

Il progetto non richiede secrets né chiavi API: la classificazione e le
checklist sono regole deterministiche definite in codice, senza chiamate a
LLM.

## Avvio in locale

```bash
pip install -r requirements.txt
streamlit run app.py
```



# Portfolio E-Commerce Sales - Dashboard vendite

Progetto dimostrativo che parte da un CSV di vendite e-commerce (valori mancanti, formati di data misti, duplicati, testo inconsistente)
e arriva a una dashboard interattiva con analisi statistica e insight di
business. Fa parte di un portfolio Streamlit multi-pagina.

## Cosa mostra il progetto

Il dataset di partenza contiene i tipici problemi di un export reale:
prezzi scritti come testo, quantità negative, date in
due formati diversi, righe duplicate per errore di doppia esportazione,
totali che non tornano con quantità * prezzo. Il progetto copre l'intera
pipeline, dalla pulizia dei dati fino alla visualizzazione:

1. **Cleaning**: normalizzazione, correzione e validazione dei dati grezzi
2. **Analisi**: KPI di business, statistiche descrittive, correlazioni
3. **Dashboard**: grafici interattivi filtrabili per periodo, categoria e stato ordine

## Architettura

```
resources/messy_ecommerce_sales_data.csv   dataset grezzo
        |
        v
eta_dashboard/cleaning.py           pipeline di pulizia (12+ step)
        |
        v
eta_dashboard/data_handle.py        DataFrameHandle: stato dati + filtri
        |
        v
eta_dashboard/analysis.py           KPI, aggregazioni, statistiche
        |
        v
eta_dashboard/view_sidebar.py       filtri (periodo, categoria, stato)
eta_dashboard/view_panels.py        3 tab: Business, Statistiche, Dati
        |
        v
pages/2_ETA_Market.py            entry point della pagina Streamlit
```

## La pipeline di cleaning, passo per passo

`clean_dataset` in `cleaning.py` esegue in ordine:

- Normalizza nomi colonna e verifica che quelle richieste siano presenti
- Pulisce testo nelle colonne categoriche (trim, case, valori vuoti -> NaN)
- Converte prezzi testuali in numeri
- Imputa `quantity` mancante con 1
- Corregge quantità e totali negativi (valore assoluto)
- Ricalcola `total` da `quantity * price` dove il valore dichiarato è incoerente o mancante, e scarta le righe non recuperabili
- Fa il parsing delle date nei due formati presenti nel dataset
- Rimuove duplicati sullo stesso `order_id`
- Marca (senza eliminare) gli outlier su quantity/price con il metodo IQR
- Scarta le righe senza campi essenziali
- Aggiunge colonne derivate (`year_month`, `weekday`) per le analisi temporali

Il risultato è un dataframe pulito più un report con le metriche prima/dopo, consultabile nel tab Dati.

## Filtri e stato

`DataFrameHandle` tiene insieme dataframe grezzo, dataframe pulito e una
maschera booleana per i filtri correnti. La sidebar scrive i valori scelti
in `st.session_state` come bozza; solo al click su "Applica" la maschera
viene ricalcolata e i pannelli si aggiornano. "Reset" riporta tutto allo
stato iniziale (nessun filtro).

## I tre tab della dashboard

- **Business Insights**: KPI (fatturato, ordini, valore medio, clienti
  unici, tasso reso/cancellazione), andamento nel tempo per status,
  fatturato per categoria, top prodotti e clienti, pattern settimanale,
  metodo di pagamento
- **Analisi statistica**: media/mediana/deviazione standard/skewness sul
  totale ordine, distribuzione e boxplot per categoria, andamento ordini
  per stato, matrice di correlazione, tabella outlier rilevati
- **Dati**: dataset pulito filtrato (scaricabile in CSV) oppure dataset
  grezzo originale, per confronto

## Configurazione

Le variabili richieste vanno messe in `.streamlit/secrets.toml`:

```toml
RES_DIR = "resources"
ETA_DASHBOARD_CSV = "messy_ecommerce_sales_data.csv"
```

## Avvio in locale

```bash
pip install -r requirements.txt
streamlit run app.py
```
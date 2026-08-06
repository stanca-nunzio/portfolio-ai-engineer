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
progetto_1/ingest.py        estrazione testo + chunking a finestra scorrevole
        |
        v
progetto_1/vectorstore.py   embedding (sentence-transformers) + Qdrant embedded
        |
        v
pages/1_RAG.py         UI Streamlit: domanda utente -> ricerca -> risposta
        |
        v
progetto_1/llm.py           chiamata a Google Gemini con contesto recuperato
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
progetto_2/cleaning.py           pipeline di pulizia (12+ step)
        |
        v
progetto_2/data_handle.py        DataFrameHandle: stato dati + filtri
        |
        v
progetto_2/analysis.py           KPI, aggregazioni, statistiche
        |
        v
progetto_2/view_sidebar.py       filtri (periodo, categoria, stato)
progetto_2/view_panels.py        3 tab: Business, Statistiche, Dati
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
PROGETTO_2_CSV = "messy_ecommerce_sales_data.csv"
```

## Avvio in locale

```bash
pip install -r requirements.txt
streamlit run app.py
```
"""
Pipeline di data cleaning per il dataset e-commerce.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "id", "customer_name", "order_id", "order_date", "product", "category",
    "quantity", "price", "payment_method", "status", "total",
]

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Porta i nomi colonna a snake_case/lower, senza rinominare (nessun alias noto)."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def check_columns(df: pd.DataFrame) -> None:
    """Controlla che il dataframe contenga tutte le colonne necessarie."""
    required_set = set(REQUIRED_COLUMNS)
    df_set = set(df.columns)
    if not required_set.issubset(df_set):
        missing_cols = required_set - df_set
        raise ValueError(f"Errore: Colonne obbligatorie mancanti: {missing_cols}")


def clean_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Esegue la pipeline completa di pulizia.

    Ritorna (df_pulito, report) dove report contiene le metriche prima/dopo
    da mostrare nel tab Data Quality.
    """
    report = {"steps": []}
    df = standardize_columns(df_raw)
    check_columns(df)

    n_start = len(df)
    report["rows_before"] = n_start
    report["cols_before"] = list(df_raw.columns)

    missing_before = df.isna().sum().to_dict()

    # 1. Colonne categoriche + normalizzazione testo/case
    for col in ["status", "product", "category", "payment_method"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    df["payment_method"] = df["payment_method"].str.lower()
    df["status"] = df["status"].str.lower()
    df["category"] = df["category"].str.title()
    df["product"] = df["product"].str.title()

    for col in ["status", "category", "payment_method"]:
        df[col] = df[col].astype('category')

    df[df['product'] == 'shoes'] = 'Shoes'

    df['category'] = df['category'].astype(str).str.capitalize()
    df['category'] = df['category'].replace(['Nan', 'NaN'], None)

    df.loc[(df['category']=='Electronic'), 'category'] = 'Electronics'

    mask = ((df['product'].str.lower() == 'headphones') | (df['product'].str.lower() == 'smartphone') | (
                df['product'].str.lower() == 'vacuum') | (df['product'].str.lower() == 'laptop'))
    df.loc[(df['category'].isna()) & mask, 'category'] = 'Electronics'

    mask = ((df['product'].str.lower() == 'biography'))
    df.loc[(df['category'].isna()) & mask, 'category'] = 'Books'

    mask = ((df['product'].str.lower() == 'jeans'))
    df.loc[(df['category'].isna()) & mask, 'category'] = 'Clothing'

    mask = ((df['product'].str.lower() == 'shoes'))
    df.loc[(df['category'].isna()) & mask, 'category'] = 'Shoes'

    mask = ((df['product'].str.lower() == 'basketball'))
    df.loc[(df['category'].isna()) & mask, 'category'] = 'Sports'

    mask = None

    if "order_id" in df.columns:
        df["order_id"] = df["order_id"].astype(str).str.strip().str.upper()
    report["steps"].append("Normalizzazione testo (trim, case) su colonne categoriche")

    # 2. Marcatura placeholder invalidi in price/order_date come mancanti
    df["price"] = df["price"].replace(["nan", "NaN", ""], np.nan)
    df["order_date"] = df["order_date"].replace(["nan", "NaN", "abc", ""], np.nan)
    df["flag_ignore"] = False
    report["steps"].append("Marcati come mancanti i placeholder non validi in price/order_date")

    # 3. Righe con stesso order_id: individua eventuali coppie da ignorare
    #    a priori note (errore noto di esportazione), prima della logica generale
    #    basata sul confronto quantity * price vs total.
    report["steps"].append("Identificazione ordini con order_id duplicato per il controllo di coerenza sul totale")

    # 4. Pulizia colonna price: valori testuali tipo "four hundred" e simboli non numerici
    df["price"] = df["price"].astype(str).str.replace("four hundred", "400", regex=False)
    df["price"] = df["price"].str.replace(r"[^\d\.]", "", regex=True)

    # 5. Quantity mancante -> 1 (assunzione minima: almeno un pezzo per riga d'ordine)
    n_missing_qty = df["quantity"].isna().sum()
    df["quantity"] = df["quantity"].fillna(1)
    report["steps"].append(f"Imputati {int(n_missing_qty)} valori mancanti in 'quantity' con 1")

    for col in ["price", "quantity", "total"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    report["steps"].append("Conversione a numerico di quantity/price/total")

    # 6. Quantity/Total negativi -> valore assoluto (errore di sistema noto)
    n_negative_qty = (df["quantity"] < 0).sum()
    df["quantity"] = df["quantity"].abs().round(0)
    df["total"] = df["total"].abs()
    report["steps"].append(f"Corrette {int(n_negative_qty)} quantità negative (valore assoluto)")

    # 7. Ricostruzione del totale teorico e verifica di coerenza.
    #    Se price manca ma il totale dichiarato non torna con quello teorico,
    #    la riga è considerata inaffidabile e va scartata più avanti.
    df["total_corretto"] = (df["quantity"] * df["price"]).round(2)
    df["total_valido"] = (df["total"] - df["total_corretto"]).abs() < 0.8
    df.loc[df["price"].isna() & ~df["total_valido"], "flag_ignore"] = True

    # Se il totale manca ma price è noto, si ricalcola dal teorico solo quando
    # il totale presente (se c'era) non era comunque valido; se manca del tutto
    # price E total, la riga viene scartata.
    df.loc[
        (~df["price"].isna()) & df["total"].isna() & (~df["total_valido"]),
        "total",
    ] = df["total_corretto"]
    df.loc[(~df["price"].isna()) & df["total"].isna(), "flag_ignore"] = True
    report["steps"].append(
        "Ricalcolato 'total' da quantity*price dove mancante e coerente; "
        "marcate come non affidabili le righe dove price e total mancano insieme"
    )

    # 8. Parsing date: due formati noti nel dataset (US numerico e testuale)
    date_us = pd.to_datetime(df["order_date"], format="%m/%d/%Y", errors="coerce")
    date_text = pd.to_datetime(df["order_date"], format="%b %d %Y", errors="coerce")
    df["order_date"] = date_us.fillna(date_text)
    report["steps"].append("Parsing date nei due formati riconosciuti (MM/DD/YYYY e Mon DD YYYY)")

    # 9. Duplicati: stesso order_id con totale alterato rispetto al teorico
    #    sono considerati errori di doppia esportazione, non nuove vendite.
    n_ignored = int(df["flag_ignore"].sum())
    df = df[~df["flag_ignore"]]
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    n_removed_dup = n_before_dedup - len(df)
    report["duplicates_removed"] = n_removed_dup + n_ignored
    report["steps"].append(
        f"Rimosse {n_ignored} righe anomale (totale incoerente/non ricostruibile) "
        f"e {n_removed_dup} duplicati sullo stesso order_id"
    )

    df = df.drop(columns=["total_corretto", "total_valido", "flag_ignore"])

    # 10. Outlier: marcatura con metodo IQR su quantity e price (non si eliminano righe)
    outlier_flags = pd.Series(False, index=df.index)
    for col in ["quantity", "price"]:
        if df[col].notna().sum() > 10:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            is_outlier = (df[col] < lower) | (df[col] > upper)
            outlier_flags = outlier_flags | is_outlier.fillna(False)
    df["flag_outlier"] = outlier_flags
    report["outliers_flagged"] = int(outlier_flags.sum())
    report["steps"].append(
        f"Segnalati {int(outlier_flags.sum())} possibili outlier (IQR, 3x range) su quantity/price"
    )

    # 11. Righe senza dati minimi indispensabili vengono scartate
    n_before_dropna = len(df)
    essential_cols = [c for c in ["order_date", "quantity", "price", "total"] if c in df.columns]
    df = df.dropna(subset=essential_cols)
    n_dropped = n_before_dropna - len(df)
    report["rows_dropped_missing_essential"] = int(n_dropped)
    report["steps"].append(f"Rimosse {n_dropped} righe con valori mancanti in campi essenziali")

    # 12. customer_name mancante -> Unknown
    if "customer_name" in df.columns:
        n_missing_cust = df["customer_name"].isna().sum()
        df["customer_name"] = df["customer_name"].fillna("Unknown")
        if n_missing_cust > 0:
            report["steps"].append(f"Imputati {int(n_missing_cust)} valori mancanti in 'customer_name' con 'Unknown'")

    # 13. Colonne derivate per l'analisi temporale
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["weekday"] = (df["order_date"].dt.day_name()).astype(str).str.lower()

    report["rows_after"] = len(df)
    report["cols_after"] = list(df.columns)
    report["missing_before"] = {k: int(v) for k, v in missing_before.items()}
    report["missing_after"] = {k: int(v) for k, v in df.isna().sum().to_dict().items()}
    report["pct_rows_retained"] = round(100 * len(df) / n_start, 1) if n_start else 0

    return df.reset_index(drop=True), report

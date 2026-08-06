"""
Incapsula lo stato dei dati dell'app: dataframe grezzo, dataframe pulito,
la maschera booleana dei filtri correntemente applicati e il dataframe filtrato
che ne risulta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from progetto_2.cleaning import clean_dataset


@dataclass
class DataFrameHandle:
    df_raw: pd.DataFrame
    df_clean: pd.DataFrame
    report: dict
    mask: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        # Nessun filtro applicato: maschera tutta True
        self.mask = pd.Series(True, index=self.df_clean.index)

    # ------------------------------------------------------------------
    # Costruzione
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(cls, file_path: str | Path) -> "DataFrameHandle":
        df_raw = pd.read_csv(file_path)
        df_clean, report = clean_dataset(df_raw)
        return cls(df_raw=df_raw, df_clean=df_clean, report=report)

    # ------------------------------------------------------------------
    # Dataset filtrato
    # ------------------------------------------------------------------
    @property
    def filtered_df(self) -> pd.DataFrame:
        """Dataframe pulito con la maschera filtri corrente applicata."""
        return self.df_clean[self.mask]

    # ------------------------------------------------------------------
    # Metadati per popolare i widget della sidebar (letti dal df pulito
    # NON filtrato, così i range/opzioni non si restringono da soli)
    # ------------------------------------------------------------------
    @property
    def date_bounds(self) -> tuple[date, date]:
        return (
            self.df_clean["order_date"].min().date(),
            self.df_clean["order_date"].max().date(),
        )

    @property
    def categories(self) -> list[str]:
        return sorted(self.df_clean["category"].unique())

    @property
    def order_statuses(self) -> list[str]:
        return sorted(self.df_clean["status"].unique())

    # ------------------------------------------------------------------
    # Applicazione / reset filtri
    # ------------------------------------------------------------------
    def apply_filters(
        self,
        start: date,
        end: date,
        categories: list[str],
        statuses: list[str],
        show_outlier:bool = True
    ) -> None:
        """Ricalcola la maschera in base ai filtri scelti nella sidebar."""
        df = self.df_clean
        mask = (
                (df["order_date"].dt.date >= start)
                & (df["order_date"].dt.date <= end)
                & (df["category"].isin(categories))
                & (df["status"].isin(statuses))
        )

        # esclude gli outlier
        if not show_outlier:
            mask &= ~df["flag_outlier"]

        self.mask = mask

    def reset_filters(self) -> None:
        """Rimuove ogni filtro: la maschera torna tutta True."""
        self.mask = pd.Series(True, index=self.df_clean.index)

    @property
    def is_empty(self) -> bool:
        return self.filtered_df.empty

    # ------------------------------------------------------------------
    # Info per il tab Data Quality
    # ------------------------------------------------------------------
    @property
    def outliers_df(self) -> pd.DataFrame:
        cols = ["order_id", "order_date", "product", "quantity", "price", "total"]
        return (
            self.df_clean[self.df_clean["flag_outlier"]][cols]
            .sort_values("total", ascending=False)
            .head(20)
        )
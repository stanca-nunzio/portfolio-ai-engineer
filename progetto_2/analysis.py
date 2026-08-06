"""
Funzioni di analisi statistica sul dataset pulito.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

ORDER_STATUS_COMPLETED = 'delivered'

def stats_completed_orders(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["status"] ==ORDER_STATUS_COMPLETED] if "status" in df.columns else df

    return completed

def kpi_summary(df: pd.DataFrame) -> dict:
    """KPI principali per le card in cima alla dashboard."""
    completed = stats_completed_orders(df)
    processing = df[df["status"] =='processing'] if "status" in df.columns else df
    return {
        "total_revenue": completed["total"].sum(),
        "total_orders": completed["order_id"].nunique(),
        "total_processing": processing["order_id"].nunique(),
        "avg_order_value": completed.groupby("order_id")["total"].sum().mean() if len(completed) else 0,
        "unique_customers": completed["customer_name"].nunique(),
        "return_rate_pct": (
            100 * (df["status"] == "returned").sum() / len(df) if "status" in df.columns and len(df) else 0
        ),
        "cancel_rate_pct": (
            100 * (df["status"] == "cancelled").sum() / len(df) if "status" in df.columns and len(df) else 0
        ),
    }


def total_by_month(df: pd.DataFrame, only_completed: bool = True) -> pd.DataFrame:
    if only_completed:
        orders = df[df["status"] == ORDER_STATUS_COMPLETED]
        group_cols = ["year_month"]
    else:
        orders = df
        group_cols = ["year_month", "status"] # Raggruppa anche per status

    return (
        orders.groupby(group_cols, as_index=False)
        .agg(total=("total", "sum"), orders=("order_id", "nunique"))
        .sort_values("year_month")
    )



def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    completed = stats_completed_orders(df)
    return (
        completed.groupby("category", as_index=False)
        .agg(revenue=("total", "sum"), Units=("quantity", "sum"), orders=("order_id", "nunique"))
        .sort_values("revenue", ascending=False)
    )


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    completed = stats_completed_orders(df)
    return (
        completed.groupby("product", as_index=False)
        .agg(revenue=("total", "sum"), Units=("quantity", "sum"))
        .sort_values("revenue", ascending=False)
        .head(n)
    )


def top_customers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    completed = stats_completed_orders(df)
    return (
        completed.groupby("customer_name", as_index=False)
        .agg(revenue=("total", "sum"), orders=("order_id", "nunique"))
        .sort_values("revenue", ascending=False)
        .head(n)
    )


def payment_method_split(df: pd.DataFrame) -> pd.DataFrame:
    completed = stats_completed_orders(df)
    return (
        completed.groupby("payment_method", as_index=False)
        .agg(orders=("order_id", "nunique"), revenue=("total", "sum"))
        .sort_values("revenue", ascending=False)
    )


def order_status_split(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("status", as_index=False)
        .agg(orders=("order_id", "nunique"))
        .sort_values("orders", ascending=False)
    )


def weekday_pattern(df: pd.DataFrame) -> pd.DataFrame:
    completed = stats_completed_orders(df)
    order = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    out = (
        completed.groupby("weekday", as_index=False)
        .agg(revenue=("total", "sum"), orders=("order_id", "nunique"))
    )
    out["weekday"] = pd.Categorical(out["weekday"], categories=order, ordered=True)
    return out.sort_values("weekday")


def revenue_stats(df: pd.DataFrame) -> dict:
    """Statistiche descrittive sul Total, per la tab statistica."""
    completed = stats_completed_orders(df)
    completed = completed["total"]
    return {
        "mean": completed.mean(),
        "median": completed.median(),
        "std": completed.std(),
        "min": completed.min(),
        "max": completed.max(),
        "q1": completed.quantile(0.25),
        "q3": completed.quantile(0.75),
        "skew": completed.skew(),
    }
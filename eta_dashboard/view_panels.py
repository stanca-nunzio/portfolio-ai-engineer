"""
Panels
======
Ogni funzione `render_*` è una view: riceve l'handle e disegna il proprio
tab leggendo SEMPRE `handle.filtered_df` (mai un df passato a parte), così
dopo un refresh vede automaticamente i dati aggiornati dai filtri.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np


from eta_dashboard.analysis import (
    kpi_summary, total_by_month, revenue_by_category, top_products,
    top_customers, payment_method_split, order_status_split,
    weekday_pattern, revenue_stats, stats_completed_orders
)
from eta_dashboard.data_handle import DataFrameHandle

PRIMARY = "#2A6F97"
ACCENT = "#F2A65A"
PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQUENCE = ["#2A6F97", "#F2A65A", "#61A0AF", "#E76F51", "#7C9885", "#A8DADC", "#4A5859"]

# Mappatura colori specifica per ogni status dell'ordine
# Usata solo nei grafici che hanno "status" come dimensione di colore
STATUS_COLORS = {
    'delivered': '#2E8B57',      # Verde - ordine completato
    'shipped': '#4682B4',        # Blu - ordine in spedizione
    'processing': '#FFA500',     # Arancione - in elaborazione
    'cancelled': '#DC143C',      # Rosso - ordine annullato
    'returned': '#9932CC',       # Viola - ordine restituito
    'pending': '#FFD700',        # Oro - in attesa
}


def render_business_tab(handle: DataFrameHandle) -> None:
    df_f = handle.filtered_df
    kpis = kpi_summary(df_f)

    c1, c2, c3, c4, c5, c6= st.columns(6)
    c1.metric("Fatturato totale", f"€{kpis['total_revenue']:,.0f}")
    c2.metric("Ordini completati", f"{kpis['total_orders']:,}")
    c3.metric("Ordini in lavorazione", f"{kpis['total_processing']:,}")
    c4.metric("Valore medio ordine", f"€{kpis['avg_order_value']:,.2f}")
    c5.metric("Clienti unici", f"{kpis['unique_customers']:,}")
    c6.metric("Tasso reso/cancellazione", f"{kpis['return_rate_pct'] + kpis['cancel_rate_pct']:.1f}%")

    st.markdown("")

    st.subheader("Andamento ordini nel tempo per status")
    rev_month = total_by_month(df_f, only_completed=False)
    fig = go.Figure()

    for status in rev_month["status"].unique():
        df_status = rev_month[rev_month["status"] == status]

        fig.add_trace(go.Scatter(
            x=df_status["year_month"],
            y=df_status["total"],
            mode="lines+markers",
            line=dict(color=STATUS_COLORS.get(status, PRIMARY), width=3),
            marker=dict(size=6),
            name=str(status),
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=580,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None, yaxis_title="Somma (€)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width='stretch')

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Andamento fatturato nel tempo (ordini consegnati)")
        rev_month = total_by_month(df_f)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rev_month["year_month"], y=rev_month["total"],
            mode="lines+markers", line=dict(color=PRIMARY, width=3),
            marker=dict(size=6), fill="tozeroy", fillcolor="rgba(42,111,151,0.08)",
            name="Fatturato",
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None, yaxis_title="Fatturato (€)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')

    with col_right:
        st.subheader("Stato ordini")
        status_df = order_status_split(df_f)
        fig = px.pie(
            status_df, names="status", values="orders",
            hole=0.55, color="status", color_discrete_map=STATUS_COLORS,
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
        )
        fig.update_traces(textinfo="percent", textposition="inside")
        st.plotly_chart(fig, width='stretch')

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Fatturato per categoria")
        cat_df = revenue_by_category(df_f)
        fig = px.bar(
            cat_df, x="revenue", y="category", orientation="h",
            color="revenue", color_continuous_scale=["#BFD9E6", PRIMARY],
            text_auto=".2s",
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(categoryorder="total ascending", title=None),
            xaxis_title="Fatturato (€)", coloraxis_showscale=False,
        )
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("Top 10 prodotti per fatturato")
        top_df = top_products(df_f, n=10)
        fig = px.bar(
            top_df, x="revenue", y="product", orientation="h",
            color_discrete_sequence=[ACCENT], text_auto=".2s",
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(categoryorder="total ascending", title=None),
            xaxis_title="Fatturato (€)",
        )
        st.plotly_chart(fig, width='stretch')

    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Top 10 clienti per fatturato")
        cust_df = top_customers(df_f, n=10)
        fig = px.bar(
            cust_df, x="revenue", y="customer_name", orientation="h",
            color_discrete_sequence=[PRIMARY], text_auto=".2s",
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(categoryorder="total ascending", title=None),
            xaxis_title="Fatturato (€)",
        )
        st.plotly_chart(fig, width='stretch')

    with col_d:
        st.subheader("Pattern settimanale")
        wd_df = weekday_pattern(df_f)
        wd_labels_it = {
            "monday": "lun", "tuesday": "mar", "wednesday": "mer", "thursday": "gio",
            "friday": "ven", "saturday": "sab", "sunday": "dom",
        }
        wd_df = wd_df.copy()
        wd_df["Giorno"] = wd_df["weekday"].map(wd_labels_it)
        fig = px.bar(
            wd_df, x="Giorno", y="revenue",
            color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None, yaxis_title="Fatturato (€)",
        )
        st.plotly_chart(fig, width='stretch')

    st.subheader("Metodo di pagamento")
    pay_df = payment_method_split(df_f)
    fig = px.bar(
        pay_df, x="payment_method", y="revenue", color="payment_method",
        color_discrete_sequence=COLOR_SEQUENCE, text_auto=".2s",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=340,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title=None, yaxis_title="Fatturato (€)", showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


def render_stats_tab(handle: DataFrameHandle) -> None:
    df_f = handle.filtered_df
    stats = revenue_stats(df_f)
    completed = stats_completed_orders(df_f)

    st.subheader("Statistiche descrittive sul totale per ordine completato")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Media", f"€{stats['mean']:.2f}")
    s2.metric("Mediana", f"€{stats['median']:.2f}")
    s3.metric("Deviazione standard", f"€{stats['std']:.2f}")
    s4.metric("Skewness", f"{stats['skew']:.2f}")

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown("#### Boxplot totale per status")
        fig = px.box(
            df_f,
            x="status", y="total",
            color="status", color_discrete_map=STATUS_COLORS,
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None, yaxis_title="Totale (€)", showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')

    with col_y:
        st.markdown("#### Boxplot totale per categoria")
        fig = px.box(
            completed,
            x="category", y="total",
            color="category", color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None, yaxis_title="Totale (€)", showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')


    st.subheader("Statistiche descrittive sullo status")

    col_x, col_y = st.columns(2)

    with col_x:
        rev_month = total_by_month(df_f, only_completed=False)

        fig_line = px.line(
            rev_month,
            x="year_month",
            y="orders",
            color="status",
            color_discrete_map=STATUS_COLORS,
            labels={
                "year_month": "Mese",
                "orders": "Numero Ordini",
                "status": "Stato",
                "total": "Totale (€)"
            },

            hover_data={
                "year_month": True,
                "status": True,
                "orders": True,
                "total": ":.2f"
            }
        )

        fig_line.update_traces(
            hovertemplate="<b>%{y}</b> ordini per un valore di <b>%{customdata[1]:.2f} €</b><extra></extra>"
        )

        fig_line.update_layout(
            template=PLOTLY_TEMPLATE,
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None,
            yaxis_title="N° Ordini",
            legend_title_text=None
        )

        st.plotly_chart(fig_line, width='stretch')

    with (col_y):

        orders = df_f.groupby(['status'], as_index=False
                     ).agg(total=("total", "sum"), orders=("order_id", "nunique"))

        fig_bar = px.bar(
            orders,
            x="status",
            y="orders",
            color="status",
            color_discrete_map=STATUS_COLORS,
            labels={
                "orders": "Numero Ordini",
                "status": "Stato",
                "total": "Totale (€)"
            },

            hover_data={
                "status": True,
                "orders": True,
                "total": ":.2f"
            }
        )

        fig_bar.update_traces(
            hovertemplate="<b>%{y}</b> ordini per un valore di <b>%{customdata[1]:.2f} €</b><extra></extra>"
        )

        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE,
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None,
            yaxis_title="N° Ordini",
            legend_title_text=None
        )

        st.plotly_chart(fig_bar, width='stretch')


    col_x, col_y = st.columns(2)

    with col_x:
        status_rate_by_category = (
            df_f.groupby(['category', 'status'])
            .size()
            .unstack(fill_value=0)
        )

        status_rate_by_category_pct = status_rate_by_category.div(
            status_rate_by_category.sum(axis=1), axis=0
        ).round(3) * 100

        df_plot = status_rate_by_category_pct.reset_index()
        df_long = df_plot.melt(
            id_vars=['category'],
            var_name='status',
            value_name='percentage'
        )

        ordered_cols = [c for c in STATUS_COLORS.keys() if c in status_rate_by_category_pct.columns]
        other_cols = [c for c in status_rate_by_category_pct.columns if c not in STATUS_COLORS]
        plot_cols = ordered_cols + other_cols

        fig_bar = px.bar(
            df_long,
            x='category',
            y='percentage',
            color='status',
            category_orders={'status': plot_cols},
            color_discrete_map=STATUS_COLORS,
            labels={'category': 'Categoria', 'percentage': 'Percentuale (%)', 'status': 'Stato'},
            hover_data={'category': True, 'status': True, 'percentage': ':.1f'}
        )

        fig_bar.update_traces(
            hovertemplate='<b>%{x}</b><br>%{fullData.name}: <b>%{y:.1f}%</b><extra></extra>'
        )

        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE,
            barmode='stack',
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None,
            yaxis_title='Percentuale (%)',
            legend_title_text=None
        )

        st.plotly_chart(fig_bar, width='stretch')

    with col_y:
        status_rate_by_payment = (
            df_f.groupby(['payment_method', 'status'])
            .size()
            .unstack(fill_value=0)
        )

        status_rate_by_payment_pct = status_rate_by_payment.div(
            status_rate_by_payment.sum(axis=1), axis=0
        ).round(3) * 100

        df_plot = status_rate_by_payment_pct.reset_index()
        df_long = df_plot.melt(
            id_vars=['payment_method'],
            var_name='status',
            value_name='percentage'
        )

        ordered_cols = [c for c in STATUS_COLORS.keys() if c in status_rate_by_payment_pct.columns]
        other_cols = [c for c in status_rate_by_payment_pct.columns if c not in STATUS_COLORS]
        plot_cols = ordered_cols + other_cols

        fig_bar = px.bar(
            df_long,
            x='payment_method',
            y='percentage',
            color='status',
            category_orders={'status': plot_cols},
            color_discrete_map=STATUS_COLORS,
            labels={'payment_method': 'Pagamento', 'percentage': 'Percentuale (%)', 'status': 'Stato'},
            hover_data={'payment_method': True, 'status': True, 'percentage': ':.1f'}
        )

        fig_bar.update_traces(
            hovertemplate='<b>%{x}</b><br>%{fullData.name}: <b>%{y:.1f}%</b><extra></extra>'
        )

        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE,
            barmode='stack',
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title=None,
            yaxis_title='Percentuale (%)',
            legend_title_text=None
        )

        st.plotly_chart(fig_bar, width='stretch')

    st.markdown("#### Correlazione tra variabili numeriche")
    numeric_cols = ["quantity", "price", "total"]
    corr = df_f[numeric_cols].corr(numeric_only=True)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    corr_masked = corr.mask(mask)

    fig = px.imshow(
        corr_masked, text_auto=".2f", color_continuous_scale=["#E76F51", "#FFFFFF", PRIMARY],
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    st.caption(
        "Nota: le correlazioni tra Quantity-Total e Price-Total sono attese (il totale dipende dalla quantità venduta e dal prezzo del prodotto)."
    )

    st.markdown("#### Outlier rilevati (metodo IQR)")
    st.caption(
        "Gli outlier non vengono eliminati automaticamente (potrebbero essere ordini legittimi), "
        "sono semplicemente marcati con un flag."
    )
    st.dataframe(handle.outliers_df, width='stretch', hide_index=True)

def render_data_tab(handle: DataFrameHandle) -> None:
    st.subheader("Esplora i dati")
    view = st.radio("Vista", ["Dati puliti (filtrati)", "Dati grezzi (originali)"], horizontal=True)

    if view == "Dati puliti (filtrati)":
        df_f = handle.filtered_df
        st.dataframe(df_f, width='stretch', hide_index=True)
        csv = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("Scarica dati puliti (CSV)", csv, "ecommerce_clean.csv", "text/csv")
    else:
        st.dataframe(handle.df_raw, width='stretch', hide_index=True)
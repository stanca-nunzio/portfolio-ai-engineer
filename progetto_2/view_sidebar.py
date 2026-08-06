"""
Sidebar
=======
Disegna i filtri nella sidebar e gestisce Applica/Reset.

I widget scrivono in `st.session_state` (valori "bozza", non ancora
applicati). Solo quando l'utente preme "Applica filtri" la sidebar chiama
`handle.apply_filters(...)`, e solo allora l'app fa il refresh dei pannelli.
"Reset" riporta sia i widget sia la maschera dell'handle allo stato iniziale.
"""

from __future__ import annotations

import streamlit as st

from progetto_2.data_handle import DataFrameHandle

_KEY_DATE = "filter_date_range"
_KEY_CATEGORY = "filter_categories"
_KEY_status = "filter_statuses"


class Sidebar:
    def __init__(self, handle: DataFrameHandle):
        self.handle = handle
        self._init_state()

    # ------------------------------------------------------------------
    def _init_state(self) -> None:
        """Popola session_state con i default solo la prima volta."""
        min_d, max_d = self.handle.date_bounds
        st.session_state.setdefault(_KEY_DATE, (min_d, max_d))
        st.session_state.setdefault(_KEY_CATEGORY, self.handle.categories)
        st.session_state.setdefault(_KEY_status, self.handle.order_statuses)

    # ------------------------------------------------------------------
    def render(self) -> bool:
        """
        Disegna i widget e i bottoni.
        Ritorna True se in questo run bisogna fare il refresh dei pannelli
        (cioè: è stato premuto Applica o Reset).
        """
        min_d, max_d = self.handle.date_bounds

        st.sidebar.header("Filtri")

        st.sidebar.date_input(
            "Periodo",
            min_value=min_d,
            max_value=max_d,
            key=_KEY_DATE,
        )
        st.sidebar.multiselect("Categoria", self.handle.categories, key=_KEY_CATEGORY)
        st.sidebar.multiselect("Stato ordine", self.handle.order_statuses, key=_KEY_status)

        col_apply, col_reset = st.sidebar.columns(2)
        apply_clicked = col_apply.button("Applica", width='stretch')
        reset_clicked = col_reset.button("Reset", width='stretch')

        st.sidebar.divider()
        st.sidebar.caption(
            f"Dataset: {self.handle.report['rows_before']:,} righe grezze → "
            f"{self.handle.report['rows_after']:,} righe pulite "
            f"({self.handle.report['pct_rows_retained']}% mantenute)"
        )

        if apply_clicked:
            self._apply()
            return True

        if reset_clicked:
            self._reset()
            return True

        return False

    # ------------------------------------------------------------------
    def _apply(self) -> None:
        date_range = st.session_state[_KEY_DATE]
        min_d, max_d = self.handle.date_bounds
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
        else:
            start, end = min_d, max_d

        self.handle.apply_filters(
            start=start,
            end=end,
            categories=st.session_state[_KEY_CATEGORY],
            statuses=st.session_state[_KEY_status],
        )

    def _reset(self) -> None:
        min_d, max_d = self.handle.date_bounds
        st.session_state[_KEY_DATE] = (min_d, max_d)
        st.session_state[_KEY_CATEGORY] = self.handle.categories
        st.session_state[_KEY_status] = self.handle.order_statuses
        self.handle.reset_filters()
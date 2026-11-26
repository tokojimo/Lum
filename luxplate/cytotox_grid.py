"""Streamlit component for validating point exclusions in Cytotox."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from .cytotox import CytotoxSignal


_grid = components.declare_component(
    "cytotox_grid", path=str(Path(__file__).parent / "resources/cytotox_grid"),
)


def _table(signal: CytotoxSignal) -> tuple[dict, dict[str, tuple[int, int]]]:
    data = signal.data
    keys = [repr(float(time)) for time in data.index]
    values = data.astype(object).where(data.notna(), None).values.tolist()
    locations = {}
    observed = []
    for row, readings in enumerate(values):
        for column, value in enumerate(readings):
            if value is not None:
                locations[f"{data.columns[column]}:{keys[row]}"] = (row, column)
                observed.append(value)
    return {
        "label": signal.name,
        "wells": data.columns.tolist(),
        "times": (data.index * 60).tolist(),
        "keys": keys,
        "values": values,
        "min": min(observed, default=0),
        "max": max(observed, default=0),
    }, locations


def review_exclusions(
    signals: tuple[CytotoxSignal, CytotoxSignal], *, file_id: str,
) -> tuple[tuple[CytotoxSignal, ...], int, bool]:
    """Apply only a submitted grid selection and keep measurements in the grid."""
    tables, locations = zip(*(_table(signal) for signal in signals))
    config = sha256(repr([(table["wells"], table["keys"]) for table in tables]).encode()).hexdigest()
    state_key = f"cytotox_exclusions_{file_id}"
    confirmed_key = f"{state_key}_confirmed"
    saved, last_event = st.session_state.get(state_key, ((set(), set()), None))
    packet = _grid(
        tables=tables,
        excluded=[sorted(points.intersection(cells)) for points, cells in zip(saved, locations)],
        config=config, key=f"cytotox_grid_{file_id}", default=None,
    )

    if (
        isinstance(packet, dict)
        and packet.get("config") == config
        and isinstance(packet.get("event"), str)
        and packet["event"] != last_event
    ):
        points = packet.get("points")
        if isinstance(points, list) and len(points) == len(tables) and all(
            isinstance(group, list) and all(isinstance(point, str) for point in group)
            for group in points
        ):
            saved = tuple(
                previous.difference(cells) | set(new).intersection(cells)
                for previous, new, cells in zip(saved, points, locations)
            )
            st.session_state[state_key] = (saved, packet["event"])
            st.session_state[confirmed_key] = config

    result = []
    count = 0
    for signal, points, cells in zip(signals, saved, locations):
        data = signal.data.copy()
        for point in points.intersection(cells):
            data.iat[cells[point]] = np.nan
            count += 1
        result.append(replace(signal, data=data))
    return tuple(result), count, st.session_state.get(confirmed_key) == config

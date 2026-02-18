"""
Load an ndx-spikesorting NWB file as a SortingAnalyzer and launch the GUI.

Assumes that ``create_sorting_analyzer_nwb.py`` was run first to generate the
test NWB file.

Usage:
    uv run python scripts/create_sorting_analyzer_nwb.py
    uv run python scripts/showcase_usage.py
"""

from __future__ import annotations

from pathlib import Path

from spikeinterface_gui import run_mainwindow

from nwb_to_sorting_analyzer import read_sorting_analyzer_from_nwb

nwb_path = Path(__file__).parent / "sorting_analyzer_test.nwb"

sorting_analyzer = read_sorting_analyzer_from_nwb(nwb_path)

sorting_analyzer.compute("unit_locations")

run_mainwindow(sorting_analyzer, mode="desktop")

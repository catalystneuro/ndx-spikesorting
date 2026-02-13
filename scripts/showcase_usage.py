"""
Load an ndx-spikesorting NWB file as a SortingAnalyzer and launch the GUI.

Assumes that `create_sorting_analyzer_nwb.py` was run first to generate the test NWB file.

Pipeline:
1. Open the NWB file and extract the SpikeSortingContainer
2. Load sorting via NwbSortingExtractor (reads units table automatically)
3. Load recording via NwbRecordingExtractor (from linked ElectricalSeries)
4. Build sparsity from the stored mask
5. Create an in-memory SortingAnalyzer and inject precomputed extensions
6. Compute unit_locations (required by GUI, derivable from templates + electrodes)
7. Launch spikeinterface-gui

Usage:
    uv run python scripts/create_sorting_analyzer_nwb.py
    uv run python scripts/showcase_usage.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO
from spikeinterface.core import ChannelSparsity, create_sorting_analyzer
from spikeinterface.core.sortinganalyzer import get_extension_class
from spikeinterface.extractors import NwbRecordingExtractor, NwbSortingExtractor

# ---- Step 1: Open the NWB file and extract the SpikeSortingContainer ----

nwb_path = Path(__file__).parent / "sorting_analyzer_test.nwb"

io = NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True)
nwbfile = io.read()

container = nwbfile.processing["ecephys"]["spike_sorting"]
sampling_frequency = container.sampling_frequency

# ---- Step 2: Load sorting via NwbSortingExtractor ----

sorting = NwbSortingExtractor(
    file_path=str(nwb_path),
    sampling_frequency=sampling_frequency,
    t_start=0.0,
)

# ---- Step 3: Load recording from linked ElectricalSeries ----

es = container.source_electrical_series
es_path = f"acquisition/{es.name}"
recording = NwbRecordingExtractor(file_path=str(nwb_path), electrical_series_path=es_path)

# ---- Step 4: Build sparsity from mask ----

sparsity = None
if container.sparsity_mask is not None:
    sparsity = ChannelSparsity(
        mask=np.array(container.sparsity_mask[:], dtype=bool),
        unit_ids=sorting.unit_ids,
        channel_ids=recording.channel_ids,
    )

# ---- Step 5: Create in-memory SortingAnalyzer ----

sorting_analyzer = create_sorting_analyzer(
    sorting=sorting,
    recording=recording,
    format="memory",
    sparse=False,
    sparsity=sparsity,
)

# ---- Step 6: Inject precomputed extensions from NWB ----

extensions = container.spike_sorting_extensions
num_channels = sorting_analyzer.get_num_channels()

# Inject random_spikes
random_spikes_nwb = extensions.random_spikes
if random_spikes_nwb is not None:
    indices_data = random_spikes_nwb.random_spikes_indices.data[:]
    index_boundaries = random_spikes_nwb.random_spikes_indices_index.data[:]

    spike_vector = sorting.to_spike_vector()
    unit_ids = sorting_analyzer.unit_ids
    global_indices = []

    for unit_index, unit_id in enumerate(unit_ids):
        start = 0 if unit_index == 0 else index_boundaries[unit_index - 1]
        end = index_boundaries[unit_index]
        unit_spike_train_indices = indices_data[start:end]

        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
        selected_samples = spike_train[unit_spike_train_indices]

        unit_mask = spike_vector["unit_index"] == unit_index
        unit_global_indices = np.where(unit_mask)[0]
        unit_samples_in_vector = spike_vector["sample_index"][unit_global_indices]

        for sample in selected_samples:
            match = np.where(unit_samples_in_vector == sample)[0]
            if len(match) > 0:
                global_indices.append(unit_global_indices[match[0]])

    random_spikes_indices = np.array(sorted(global_indices), dtype=np.int64)

    ext_class = get_extension_class("random_spikes")
    ext = ext_class(sorting_analyzer)
    ext.params = {"method": "uniform", "max_spikes_per_unit": 500, "margin_size": None, "seed": None}
    ext.data["random_spikes_indices"] = random_spikes_indices
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["random_spikes"] = ext

# Inject templates
templates_nwb = extensions.templates
if templates_nwb is not None:
    sparse_data = templates_nwb.data.data[:]  # (num_waveforms, num_samples)
    data_index = templates_nwb.data_index.data[:]
    electrode_indices = templates_nwb.electrodes.data[:]
    peak_sample_index = templates_nwb.peak_sample_index

    num_units = len(sorting_analyzer.unit_ids)
    num_samples = sparse_data.shape[1]

    # nbefore + nafter = num_samples, peak_sample_index == nbefore
    ms_before = peak_sample_index / sampling_frequency * 1000.0
    ms_after = (num_samples - peak_sample_index) / sampling_frequency * 1000.0

    dense_templates = np.zeros((num_units, num_samples, num_channels), dtype=np.float32)
    for unit_index in range(num_units):
        start = 0 if unit_index == 0 else data_index[unit_index - 1]
        end = data_index[unit_index]
        unit_sparse = sparse_data[start:end, :]
        active_channels = electrode_indices[start:end]

        for i, ch in enumerate(active_channels):
            dense_templates[unit_index, :, ch] = unit_sparse[i, :]

    ext_class = get_extension_class("templates")
    ext = ext_class(sorting_analyzer)
    ext.params = {"ms_before": float(ms_before), "ms_after": float(ms_after), "operators": ["average", "std"]}
    ext.data["average"] = dense_templates
    ext.data["std"] = np.zeros_like(dense_templates)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["templates"] = ext

# ---- Step 7: Compute unit_locations and launch GUI ----

sorting_analyzer.compute("unit_locations")

from spikeinterface_gui import run_mainwindow

run_mainwindow(sorting_analyzer, mode="desktop")

io.close()

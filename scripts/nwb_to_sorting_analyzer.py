"""Reusable loader: NWB file with ndx-spikesorting data -> SortingAnalyzer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pynwb import NWBHDF5IO
from spikeinterface.core import ChannelSparsity, create_sorting_analyzer
from spikeinterface.core.sortinganalyzer import get_extension_class
from spikeinterface.extractors import NwbRecordingExtractor, NwbSortingExtractor

if TYPE_CHECKING:
    from spikeinterface.core import SortingAnalyzer


def read_sorting_analyzer_from_nwb(nwbfile_path: str | Path) -> SortingAnalyzer:
    """Read an ndx-spikesorting NWB file and return a SortingAnalyzer.

    Opens the NWB file, extracts the SpikeSortingContainer from
    ``processing["ecephys"]["spike_sorting"]``, reconstructs sorting/recording
    objects, and instantiates any precomputed extensions (random_spikes, templates).

    Parameters
    ----------
    nwbfile_path : str or Path
        Path to the NWB file on disk.

    Returns
    -------
    SortingAnalyzer
        In-memory SortingAnalyzer with precomputed extensions instantiated.
    """
    nwbfile_path = Path(nwbfile_path)

    # -- Open NWB and locate the container --
    io = NWBHDF5IO(nwbfile_path, mode="r", load_namespaces=True)
    nwbfile = io.read()

    container = nwbfile.processing["ecephys"]["spike_sorting"]
    sampling_frequency = container.sampling_frequency

    # -- Load sorting --
    sorting = NwbSortingExtractor(
        file_path=nwbfile_path,
        sampling_frequency=sampling_frequency,
        t_start=0.0,
    )

    # -- Load recording (if linked) --
    recording = None
    if container.source_electrical_series is not None:
        es = container.source_electrical_series
        es_path = f"acquisition/{es.name}"
        recording = NwbRecordingExtractor(file_path=nwbfile_path, electrical_series_path=es_path)

    # -- Build sparsity from stored mask --
    sparsity = None
    if container.sparsity_mask is not None:
        channel_ids = recording.channel_ids if recording is not None else np.array(container.electrodes.data[:])
        sparsity = ChannelSparsity(
            mask=np.array(container.sparsity_mask[:], dtype=bool),
            unit_ids=sorting.unit_ids,
            channel_ids=channel_ids,
        )

    # -- Create in-memory SortingAnalyzer --
    sorting_analyzer = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="memory",
        sparse=False,
        sparsity=sparsity,
    )

    # -- Instantiate precomputed extensions --
    extensions = container.spike_sorting_extensions
    if extensions is not None:
        _load_random_spikes_extension_from_nwb(extensions, sorting, sorting_analyzer)
        _load_templates_extension_from_nwb(extensions, sorting_analyzer, sampling_frequency)

    return sorting_analyzer


def _load_random_spikes_extension_from_nwb(extensions, sorting, sorting_analyzer):
    """Instantiate the random_spikes extension if present in the NWB container.

    This requires converting between two different index representations:

    **SpikeInterface** stores a single flat array of global indices into the
    spike vector (all units interleaved by time)::

        random_spikes_indices: [3, 7, 15, 22, 41, 58, 63, ...]
                                ^                    ^
                                index into sorting.to_spike_vector()

    Index 3 means "the 4th spike overall (across all units, in time order)".
    There is no per-unit grouping.

    **NWB** stores a per-unit ragged array of indices into each unit's spike
    train. The two datasets work together:

    ``random_spikes_indices_index`` is a VectorIndex that stores cumulative
    boundary positions, one value per unit. Each value marks where that
    unit's slice ends in ``random_spikes_indices``::

        random_spikes_indices_index:  [3,          6,           8       ]
                                       ^           ^            ^
                                       unit 0      unit 1       unit 2
                                       ends at 3   ends at 6    ends at 8

    These boundaries partition ``random_spikes_indices`` into per-unit
    segments. Each segment contains local indices into that unit's own
    spike train::

        random_spikes_indices:  [2,  5,  8,  |  1,  4,  12,  |  0,  3  ]
                                 \\_________/     \\__________/     \\____/
                                   unit 0          unit 1         unit 2
                                  [0:3]           [3:6]           [6:8]
                                    ^               ^               ^
                                 3rd, 6th,       2nd, 5th,       1st, 4th
                                 9th spike       13th spike      spike of
                                 of unit 0       of unit 1       unit 2

    Each value is a local index. ``2`` means "the 3rd spike of that
    unit", not the 3rd spike overall.

    The conversion maps each per-unit local index back to the global spike
    vector by matching sample times.
    """
    from spikeinterface.core.sorting_tools import spike_vector_to_indices

    random_spikes_nwb = extensions.random_spikes
    if random_spikes_nwb is None:
        return

    indices_data = random_spikes_nwb.random_spikes_indices.data[:]
    index_boundaries = random_spikes_nwb.random_spikes_indices_index.data[:]

    # -- Infer params from stored data --
    unit_ids = sorting_analyzer.unit_ids
    per_unit_counts = np.diff(np.concatenate([[0], index_boundaries]))

    total_spike_counts = sorting.count_num_spikes_per_unit(outputs="array")
    if np.array_equal(per_unit_counts, total_spike_counts):
        method = "all"
    else:
        method = "uniform"
    max_spikes_per_unit = int(per_unit_counts.max())

    # -- Convert per-unit spike train indices to global spike vector indices --
    spike_vector = sorting.to_spike_vector(concatenated=False)
    spike_indices_by_unit_and_segments = spike_vector_to_indices(spike_vector, unit_ids, absolute_index=True)
    # TODO: deal with multi-segment case if it arises in the future. For now we assume single segment
    segment_index = 0
    spike_indices_by_unit = spike_indices_by_unit_and_segments[segment_index]
    
    global_indices = []

    for unit_index, unit_id in enumerate(unit_ids):
        start = 0 if unit_index == 0 else index_boundaries[unit_index - 1]
        end = index_boundaries[unit_index]
        unit_spike_train_indices = indices_data[start:end]
        global_indices.extend(spike_indices_by_unit[unit_id][unit_spike_train_indices])

    random_spikes_indices = np.array(sorted(global_indices), dtype=np.int64)

    ext_class = get_extension_class("random_spikes")
    ext = ext_class(sorting_analyzer)
    ext.set_params(method=method, max_spikes_per_unit=max_spikes_per_unit)
    ext.data["random_spikes_indices"] = random_spikes_indices
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["random_spikes"] = ext


def _load_templates_extension_from_nwb(extensions, sorting_analyzer, sampling_frequency):
    """Instantiate the templates extension if present in the NWB container.

    This requires converting between two different template representations:

    **NWB** stores templates as a sparse ragged array, only active channels
    per unit, concatenated across units::

        data:       [[ch2_samples, ch5_samples, ch7_samples],  # unit 0 (3 active channels)
                     [ch1_samples, ch5_samples],                # unit 1 (2 active channels)
                     ...]
        data_index: [3, 5, ...]          # cumulative row boundaries per unit
        electrodes: [2, 5, 7, 1, 5, ...]  # which channel each row belongs to

    This is compact (no zeros for inactive channels) and tool-agnostic.

    **SpikeInterface** stores templates as a 3D array with shape
    ``(n_units, n_samples, n_channels)``. When a sparsity mask is set,
    ``n_channels`` is ``max_num_active_channels`` (the maximum number of
    active channels across all units), not the total channel count. Each
    unit's data is packed into the first N columns corresponding to its
    active channels, and units with fewer active channels have trailing
    zeros. At the end of ``ComputeTemplates._run``,
    ``sparsity.densify_templates()`` expands this compact representation
    into a fully dense array of shape
    ``(n_units, n_samples, total_channels)`` before assigning it to
    ``ext.data["average"]`` in memory. So the in-memory representation
    always has the full channel count. The sparsity information lives
    separately in the ``ChannelSparsity`` object on the analyzer, and
    methods like ``get_unit_template()`` use it at access time to return
    only the relevant channels for a given unit.

    The conversion scatters each sparse row into the correct channel position
    of the dense array.
    """
    from ndx_spikesorting import templates_to_dense
    templates_nwb = extensions.templates
    if templates_nwb is None:
        return

    peak_sample_index = templates_nwb.peak_sample_index
    num_channels = sorting_analyzer.get_num_channels()
    num_samples = templates_nwb.data.data.shape[1]

    # Derive timing params from peak_sample_index and sampling_frequency
    ms_before = peak_sample_index / sampling_frequency * 1000.0
    ms_after = (num_samples - peak_sample_index) / sampling_frequency * 1000.0

    # Reconstruct dense templates from sparse ragged arrays
    dense_templates = templates_to_dense(templates_nwb, num_channels)

    ext_class = get_extension_class("templates")
    ext = ext_class(sorting_analyzer)
    ext.set_params(ms_before=float(ms_before), ms_after=float(ms_after), operators=["average", "std"])
    ext.data["average"] = dense_templates
    ext.data["std"] = np.zeros_like(dense_templates)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["templates"] = ext

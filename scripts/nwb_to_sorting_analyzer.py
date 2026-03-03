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
        _load_random_spikes_extension_from_nwb(extensions, sorting_analyzer)
        _load_templates_extension_from_nwb(extensions, sorting_analyzer)
        _load_noise_levels_extension_from_nwb(extensions, sorting_analyzer)
        _load_unit_locations_extension_from_nwb(extensions, sorting_analyzer)
        _load_correlograms_extension_from_nwb(extensions, sorting_analyzer)
        _load_isi_histograms_extension_from_nwb(extensions, sorting_analyzer)
        _load_template_similarity_extension_from_nwb(extensions, sorting_analyzer)
        _load_spike_amplitudes_extension_from_nwb(extensions, sorting_analyzer)
        _load_amplitude_scalings_extension_from_nwb(extensions, sorting_analyzer)
        _load_spike_locations_extension_from_nwb(extensions, sorting_analyzer)
        

    return sorting_analyzer


def _load_random_spikes_extension_from_nwb(extensions, sorting_analyzer):
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

    sorting = sorting_analyzer.sorting
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


def _load_templates_extension_from_nwb(extensions, sorting_analyzer):
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
    ms_before = peak_sample_index / sorting_analyzer.sampling_frequency * 1000.0
    ms_after = (num_samples - peak_sample_index) / sorting_analyzer.sampling_frequency * 1000.0

    # Reconstruct dense templates from sparse ragged arrays
    dense_templates = templates_to_dense(templates_nwb, num_channels)

    ext_class = get_extension_class("templates")
    ext = ext_class(sorting_analyzer)
    ext.set_params(ms_before=float(ms_before), ms_after=float(ms_after), operators=["average", "std"])
    ext.data["average"] = dense_templates
    ext.data["std"] = np.zeros_like(dense_templates)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["templates"] = ext


def _load_noise_levels_extension_from_nwb(extensions, sorting_analyzer):
    """Instantiate the noise_levels extension if present in the NWB container.

    The NWB extension stores a simple dense array of noise levels with shape
    (num_channels,). This maps directly to the expected format for the
    NoiseLevels extension in SpikeInterface, so no complex conversion is needed.
    Each value corresponds to a channel in the same order as sorting_analyzer.channel_ids.

    """
    noise_levels_nwb = extensions.noise_levels
    if noise_levels_nwb is None:
        return

    noise_data = noise_levels_nwb.data[:]

    ext_class = get_extension_class("noise_levels")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["noise_levels"] = noise_data.astype(np.float32)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["noise_levels"] = ext


def _load_unit_locations_extension_from_nwb(extensions, sorting_analyzer):
    """Instantiate the unit_locations extension if present in the NWB container.

    The NWB extension stores a simple dense array of unit locations with shape
    (num_units, 3) for (x, y, z) coordinates. This maps directly to the
    expected format for the UnitLocations extension in SpikeInterface, so no
    complex conversion is needed. Each row corresponds to a unit in the same order as sorting_analyzer.unit_ids.

    """
    unit_locations_nwb = extensions.unit_locations
    if unit_locations_nwb is None:
        return

    locations_data = unit_locations_nwb.data[:]

    ext_class = get_extension_class("unit_locations")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["locations"] = locations_data.astype(np.float32)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["unit_locations"] = ext


def _load_correlograms_extension_from_nwb(extensions, sorting_analyzer):
    """Instantiate the correlograms extension if present in the NWB container.

    The NWB extension stores a simple dense array of correlograms with shape
    (num_units, num_units, num_bins). This maps directly to the expected format for the
    Correlograms extension in SpikeInterface, so no complex conversion is needed. Each row/column corresponds to a unit in the same order as sorting_analyzer.unit_ids.

    """
    correlograms_nwb = extensions.correlograms
    if correlograms_nwb is None:
        return

    correlograms_data = correlograms_nwb.data[:]

    ext_class = get_extension_class("correlograms")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["ccgs"] = correlograms_data
    ext.data["bins"] = correlograms_nwb.bin_edges[:]
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["correlograms"] = ext


def _load_isi_histograms_extension_from_nwb(extensions, sorting_analyzer):
    """Instantiate the isi histograms extension if present in the NWB container.

    The NWB extension stores a simple dense array of isi histograms with shape
    (num_units, num_bins). This maps directly to the expected format for the
    ISIHistograms extension in SpikeInterface, so no complex conversion is needed. Each row/column corresponds to a unit in the same order as sorting_analyzer.unit_ids.

    """
    isi_histograms_nwb = extensions.isi_histograms
    if isi_histograms_nwb is None:
        return

    isi_histograms_data = isi_histograms_nwb.data[:]

    ext_class = get_extension_class("isi_histograms")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["isi_histograms"] = isi_histograms_data
    ext.data["bins"] = isi_histograms_nwb.bin_edges[:]
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["isi_histograms"] = ext

def _load_template_similarity_extension_from_nwb(extensions, sorting_analyzer):
    """Instantiate the template_similarity extension if present in the NWB container.

    The NWB extension stores a simple dense array of template similarity with shape
    (num_units, num_units). This maps directly to the expected format for the
    TemplateSimilarity extension in SpikeInterface, so no complex conversion is needed. Each row/column corresponds to a unit in the same order as sorting_analyzer.unit_ids.

    """
    template_similarity_nwb = extensions.template_similarity
    if template_similarity_nwb is None:
        return

    template_similarity_data = template_similarity_nwb.data[:]

    ext_class = get_extension_class("template_similarity")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["similarity"] = template_similarity_data
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["template_similarity"] = ext


def _load_spike_amplitudes_extension_from_nwb(extensions, sorting_analyzer):
    """Instantiate the spike_amplitudes extension if present in the NWB container.
    """
    spike_amplitudes_nwb = extensions.spike_amplitudes
    if spike_amplitudes_nwb is None:
        return

    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    unit_indices = spike_vector["unit_index"]
    sort_order = np.argsort(unit_indices)
    reverse_order = np.argsort(sort_order, kind="stable")
    spike_amplitudes = spike_amplitudes_nwb.data[:][reverse_order]
    ext_class = get_extension_class("spike_amplitudes")
    ext = ext_class(sorting_analyzer)
    ext.data["amplitudes"] = spike_amplitudes.astype(np.float32)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["spike_amplitudes"] = ext


def _load_spike_locations_extension_from_nwb(extensions, sorting_analyzer):
    """Instantiate the spike_locations extension if present in the NWB container.
    """
    spike_locations_nwb = extensions.spike_locations
    if spike_locations_nwb is None:
        return

    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    unit_indices = spike_vector["unit_index"]
    sort_order = np.argsort(unit_indices)
    reverse_order = np.argsort(sort_order, kind="stable")
    spike_locations = spike_locations_nwb.data[:][reverse_order]
    ext_class = get_extension_class("spike_locations")
    ext = ext_class(sorting_analyzer)
    # make x, y or x, y, z structured array
    if spike_locations.shape[1] == 2:
        spike_locations = np.core.records.fromarrays(spike_locations.T, names="x,y")
    elif spike_locations.shape[1] == 3:
        spike_locations = np.core.records.fromarrays(spike_locations.T, names="x,y,z")
    ext.set_params()
    ext.data["spike_locations"] = spike_locations
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["spike_locations"] = ext


def _load_amplitude_scalings_extension_from_nwb(extensions, sorting_analyzer):
    """Instantiate the amplitude_scalings extension if present in the NWB container.
    """
    amplitude_scalings_nwb = extensions.amplitude_scalings
    if amplitude_scalings_nwb is None:
        return

    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    unit_indices = spike_vector["unit_index"]
    sort_order = np.argsort(unit_indices)
    reverse_order = np.argsort(sort_order, kind="stable")
    amplitude_scalings = amplitude_scalings_nwb.data[:][reverse_order]
    ext_class = get_extension_class("amplitude_scalings")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["amplitude_scalings"] = amplitude_scalings.astype(np.float32)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["amplitude_scalings"] = ext
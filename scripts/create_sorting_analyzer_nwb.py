"""
Script to create a SortingAnalyzer from SpikeInterface and save it to NWB.

This script demonstrates how to:
1. Generate mock recording and sorting data using SpikeInterface
2. Create a SortingAnalyzer and compute extensions (random_spikes, templates)
3. Write recording and sorting to NWB via neuroconv
4. Add ndx-spikesorting extension data (templates, random_spikes, sparsity) on top
5. Link the SpikeSortingContainer to the ElectricalSeries for recording mode
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from hdmf.common import VectorData, VectorIndex, DynamicTableRegion
from neuroconv.tools.spikeinterface import add_recording_to_nwbfile, add_sorting_to_nwbfile
from pynwb import NWBHDF5IO, NWBFile
from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording

from ndx_spikesorting import (
    RandomSpikes,
    Waveforms,
    Templates,
    NoiseLevels,
    UnitLocations,
    Correlograms,
    ISIHistograms,
    TemplateSimilarity,
    SpikeAmplitudes,
    SpikeLocations,
    AmplitudeScalings,
    PrincipalComponents,
    SpikeSortingContainer,
    SpikeSortingExtensions,
)

# ---- Step 1: Generate mock data and create a SortingAnalyzer ----

recording, sorting = generate_ground_truth_recording(
    durations=[5.0],
    num_units=5,
    num_channels=10,
    seed=42,
)

sorting_analyzer = create_sorting_analyzer(
    sorting=sorting,
    recording=recording,
    format="memory",
    sparse=True,
)

sorting_analyzer.compute(
    {
        "random_spikes": {"max_spikes_per_unit": 10, "seed": 42},
        "waveforms": {},
        "templates": {},
        "noise_levels": {},
        "unit_locations": {"method": "monopolar_triangulation"},
        "correlograms": {},
        "principal_components": {"n_components": 3},
        "isi_histograms": {},
        "template_similarity": {},
        "spike_amplitudes": {},
        "amplitude_scalings": {},
        "spike_locations": {"method": "grid_convolution"}
    }
)

unit_ids = sorting_analyzer.unit_ids
sparsity = sorting_analyzer.sparsity
# ---- Step 2: Create the base NWB file with recording and sorting via neuroconv ----

nwbfile = NWBFile(
    session_description="SortingAnalyzer test data from SpikeInterface",
    identifier=f"sorting_analyzer_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    session_start_time=datetime.now(timezone.utc),
    experimenter=["Test User"],
    lab="NWB Extension Test",
    institution="SpikeInterface",
)

add_recording_to_nwbfile(recording, nwbfile=nwbfile, write_as="raw", iterator_type=None)
add_sorting_to_nwbfile(sorting, nwbfile=nwbfile, write_as="units")

electrical_series = nwbfile.acquisition["ElectricalSeriesRaw"]

electrodes_region = nwbfile.create_electrode_table_region(
    region=list(range(recording.get_num_channels())),
    description="All electrodes used in sorting analysis",
)

units_region = DynamicTableRegion(
    name="units_region",
    data=list(range(len(sorting.unit_ids))),
    description="All units from sorting analysis",
    table=nwbfile.units,
)

# ---- Step 3: Convert extensions to NWB ----

# Random spikes
random_spikes_ext = sorting_analyzer.get_extension("random_spikes")
random_spikes_data = random_spikes_ext.get_random_spikes()

all_indices = []
cumulative_index = []

for unit_id in unit_ids:
    spike_train = sorting.get_unit_spike_train(unit_id=unit_id)

    unit_mask = random_spikes_data["unit_index"] == list(unit_ids).index(unit_id)
    unit_spike_indices = random_spikes_data["sample_index"][unit_mask]

    unit_indices = []
    for sample_index in unit_spike_indices:
        matches = np.where(spike_train == sample_index)[0]
        if len(matches) > 0:
            unit_indices.append(matches[0])

    unit_indices = np.array(unit_indices, dtype=np.int64)
    all_indices.append(unit_indices)
    cumulative_index.append(sum(len(indices) for indices in all_indices))

random_spikes_indices = VectorData(
    name="random_spikes_indices",
    data=np.concatenate(all_indices).astype(np.int64),
    description="Random spike indices for all units",
)

random_spikes_indices_index = VectorIndex(
    name="random_spikes_indices_index",
    data=np.array(cumulative_index, dtype=np.int64),
    target=random_spikes_indices,
)

nwb_random_spikes = RandomSpikes(
    name="random_spikes",
    random_spikes_indices=random_spikes_indices,
    random_spikes_indices_index=random_spikes_indices_index,
)

# Waveforms (double-ragged: rows grouped by spike, spikes grouped by unit)
waveforms_ext = sorting_analyzer.get_extension("waveforms")

all_wf_rows = []
all_wf_electrode_indices = []
wf_spike_cumulative = []   # data_index: cumulative row count per spike
wf_unit_cumulative = []    # data_index_index: cumulative spike count per unit

total_spikes = 0
for unit_id in unit_ids:
    # (n_spikes, n_samples, n_channels_sparse)
    unit_waveforms = waveforms_ext.get_waveforms_one_unit(unit_id=unit_id, force_dense=False)
    n_spikes_unit = unit_waveforms.shape[0]

    if sparsity is not None:
        channel_indices = sparsity.unit_id_to_channel_indices[unit_id]
    else:
        channel_indices = np.arange(unit_waveforms.shape[2])

    for spike_idx in range(n_spikes_unit):
        # (n_channels, n_samples) — one row per channel
        spike_wf = unit_waveforms[spike_idx].T  # from (n_samples, n_channels) to (n_channels, n_samples)
        all_wf_rows.append(spike_wf)
        all_wf_electrode_indices.extend(channel_indices)
        wf_spike_cumulative.append(len(np.vstack(all_wf_rows)))

    total_spikes += n_spikes_unit
    wf_unit_cumulative.append(total_spikes)

wf_data = VectorData(
    name="data",
    data=np.vstack(all_wf_rows).astype(np.float32),
    description="Waveform data (one row per channel per spike)",
)

wf_data_index = VectorIndex(
    name="data_index",
    data=np.array(wf_spike_cumulative, dtype=np.int64),
    target=wf_data,
)

wf_data_index_index = VectorIndex(
    name="data_index_index",
    data=np.array(wf_unit_cumulative, dtype=np.int64),
    target=wf_data_index,
)

wf_electrodes = DynamicTableRegion(
    name="electrodes",
    data=list(int(i) for i in all_wf_electrode_indices),
    description="Electrode for each waveform row.",
    table=nwbfile.electrodes,
)

nwb_waveforms = Waveforms(
    name="waveforms",
    data=wf_data,
    data_index=wf_data_index,
    data_index_index=wf_data_index_index,
    electrodes=wf_electrodes,
)

# Templates
templates_ext = sorting_analyzer.get_extension("templates")
nbefore = templates_ext.nbefore

all_data = []
all_electrode_indices = []
cumulative_index = []

for unit_id in unit_ids:
    template = templates_ext.get_unit_template(unit_id=unit_id)  # (num_samples, num_channels)

    if sparsity is not None:
        channel_indices = sparsity.unit_id_to_channel_indices[unit_id]
        sparse_template = template[:, channel_indices].T  # (num_active_channels, num_samples)
    else:
        channel_indices = np.arange(template.shape[1])
        sparse_template = template.T  # (num_channels, num_samples)

    all_data.append(sparse_template)
    all_electrode_indices.extend(channel_indices)
    cumulative_index.append(len(np.vstack(all_data)))

data = VectorData(
    name="data",
    data=np.vstack(all_data).astype(np.float32),
    description="Template waveforms",
)

data_index = VectorIndex(
    name="data_index",
    data=np.array(cumulative_index, dtype=np.int64),
    target=data,
)

# Note: this DynamicTableRegion produces a harmless hdmf warning about not sharing
# an ancestor with the electrodes table. This is expected because Templates is
# nested deep in the container tree while the electrodes table is at the NWBFile root.
# The reference resolves correctly once written to disk.
template_electrodes = DynamicTableRegion(
    name="electrodes",
    data=list(int(i) for i in all_electrode_indices),
    description="Electrode for each waveform row.",
    table=nwbfile.electrodes,
)

nwb_templates = Templates(
    name="templates",
    peak_sample_index=int(nbefore),
    data=data,
    data_index=data_index,
    electrodes=template_electrodes,
)

# NoiseLevels
noise_levels_ext = sorting_analyzer.get_extension("noise_levels")
noise_levels_data = noise_levels_ext.get_data()
nwb_noise_levels = NoiseLevels(
    name="noise_levels",
    data=noise_levels_data,
)

# UnitLocations
unit_locations_ext = sorting_analyzer.get_extension("unit_locations")
unit_locations_data = unit_locations_ext.get_data()
nwb_unit_locations = UnitLocations(
    name="unit_locations",
    data=unit_locations_data,
)

# Correlograms
correlograms_ext = sorting_analyzer.get_extension("correlograms")
ccgs, bin_edges = correlograms_ext.get_data()
nwb_correlograms = Correlograms(
    name="correlograms",
    data=ccgs,
    bin_edges=bin_edges
)

# ISIHistograms
isi_ext = sorting_analyzer.get_extension("isi_histograms")
isis, bin_edges = isi_ext.get_data()
nwb_isi_histograms = ISIHistograms(
    name="isi_histograms",
    data=isis,
    bin_edges=bin_edges
)

# TemplateSimilarity
template_similarity_ext = sorting_analyzer.get_extension("template_similarity")
similarity = template_similarity_ext.get_data()
nwb_template_similarity = TemplateSimilarity(
    name="template_similarity",
    data=similarity,
)

# SpikeAmplitudes, AmplitudeScalings, SpikeLocations
base_vector_extensions = ["spike_amplitudes", "spike_locations", "amplitude_scalings"]
nwb_classes = {
    "spike_amplitudes": SpikeAmplitudes,
    "spike_locations": SpikeLocations,
    "amplitude_scalings": AmplitudeScalings
}
nwb_extensions = {}
spike_vector = sorting_analyzer.sorting.to_spike_vector()
unit_indices = spike_vector["unit_index"]
sort_order = np.argsort(unit_indices)
cumulative_index = np.cumsum(np.bincount(unit_indices))
for extension_name in base_vector_extensions:
    extension = sorting_analyzer.get_extension(extension_name)
    all_data = extension.get_data()[sort_order]

    if all_data.dtype.names is not None:
        all_data = np.stack([all_data[name] for name in all_data.dtype.names], axis=1)

    data = VectorData(
        name="data",
        data=all_data,
        description=f"{extension_name} data",
    )

    data_index = VectorIndex(
        name="data_index",
        data=np.array(cumulative_index, dtype=np.int64),
        target=data,
    )
    nwb_extensions[extension_name] = nwb_classes[extension_name](
        name=extension_name,
        data=data,
        data_index=data_index
    )

# PrincipalComponents (double-ragged: rows grouped by spike, spikes grouped by unit)
pc_ext = sorting_analyzer.get_extension("principal_components")
pc_mode = pc_ext.params.get("mode", "by_channel_local")

all_pc_rows = []
all_pc_electrode_indices = []
pc_spike_cumulative = []
pc_unit_cumulative = []

total_spikes = 0
for unit_id in unit_ids:
    # (n_spikes, n_components, n_channels_sparse) for by_channel_local
    # (n_spikes, n_components) for concatenated
    unit_pcs = pc_ext.get_projections_one_unit(unit_id=unit_id, sparse=True)
    if isinstance(unit_pcs, tuple):
        unit_pcs, _ = unit_pcs
    n_spikes_unit = unit_pcs.shape[0]

    if unit_pcs.ndim == 3:
        # Per-channel mode
        if sparsity is not None:
            channel_indices = sparsity.unit_id_to_channel_indices[unit_id]
        else:
            channel_indices = np.arange(unit_pcs.shape[2])

        for spike_idx in range(n_spikes_unit):
            # (n_channels, n_components)
            spike_pc = unit_pcs[spike_idx].T  # from (n_components, n_channels) to (n_channels, n_components)
            all_pc_rows.append(spike_pc)
            all_pc_electrode_indices.extend(channel_indices)
            pc_spike_cumulative.append(len(np.vstack(all_pc_rows)))
    else:
        # Concatenated mode: one row per spike
        for spike_idx in range(n_spikes_unit):
            all_pc_rows.append(unit_pcs[spike_idx:spike_idx + 1])  # (1, n_components)
            pc_spike_cumulative.append(len(np.vstack(all_pc_rows)))

    total_spikes += n_spikes_unit
    pc_unit_cumulative.append(total_spikes)

pc_data = VectorData(
    name="data",
    data=np.vstack(all_pc_rows).astype(np.float64),
    description="PCA projection data",
)

pc_data_index = VectorIndex(
    name="data_index",
    data=np.array(pc_spike_cumulative, dtype=np.int64),
    target=pc_data,
)

pc_data_index_index = VectorIndex(
    name="data_index_index",
    data=np.array(pc_unit_cumulative, dtype=np.int64),
    target=pc_data_index,
)

# Only include electrodes for per-channel mode
if unit_pcs.ndim == 3:
    pc_electrodes = DynamicTableRegion(
        name="electrodes",
        data=list(int(i) for i in all_pc_electrode_indices),
        description="Electrode for each projection row.",
        table=nwbfile.electrodes,
    )
else:
    pc_electrodes = None

nwb_principal_components = PrincipalComponents(
    name="principal_components",
    data=pc_data,
    data_index=pc_data_index,
    data_index_index=pc_data_index_index,
    electrodes=pc_electrodes,
)


# ---- Step 4: Assemble the SpikeSortingContainer and write to NWB ----

sparsity_mask = sparsity.mask if sparsity is not None else None

extensions = SpikeSortingExtensions(name="extensions")
extensions.random_spikes = nwb_random_spikes
extensions.waveforms = nwb_waveforms
extensions.templates = nwb_templates
extensions.noise_levels = nwb_noise_levels
extensions.unit_locations = nwb_unit_locations
extensions.correlograms = nwb_correlograms
extensions.isi_histograms = nwb_isi_histograms
extensions.template_similarity = nwb_template_similarity
extensions.spike_amplitudes = nwb_extensions["spike_amplitudes"]
extensions.spike_locations = nwb_extensions["spike_locations"]
extensions.amplitude_scalings = nwb_extensions["amplitude_scalings"]
extensions.principal_components = nwb_principal_components

container = SpikeSortingContainer(
    name="spike_sorting",
    sampling_frequency=sorting_analyzer.sampling_frequency,
    electrodes=electrodes_region,
    units_region=units_region,
    sparsity_mask=sparsity_mask,
    source_electrical_series=electrical_series,
)
container.spike_sorting_extensions = extensions

ecephys_module = nwbfile.create_processing_module(
    name="ecephys",
    description="Extracellular electrophysiology processing results",
)
ecephys_module.add(container)

# ---- Step 6: Write to disk ----

output_file = Path(__file__).parent / "sorting_analyzer_test.nwb"
with NWBHDF5IO(str(output_file), mode="w") as io:
    io.write(nwbfile)

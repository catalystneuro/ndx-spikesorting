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
    Templates,
    NoiseLevels,
    UnitLocations,
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
        "templates": {},
        "noise_levels": {},
        "unit_locations": {"method": "monopolar_triangulation"},
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

# ---- Step 3: Convert random_spikes extension to NWB ----

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

# ---- Step 4: Convert templates extension to NWB ----

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

# ---- Step 5: Convert noise_levels extension to NWB ----
noise_levels_ext = sorting_analyzer.get_extension("noise_levels")
noise_levels_data = noise_levels_ext.get_data()
nwb_noise_levels = NoiseLevels(
    name="noise_levels",
    data=noise_levels_data,
)

# ---- Step 6: Convert unit locations extension to NWB ----
unit_locations_ext = sorting_analyzer.get_extension("unit_locations")
unit_locations_data = unit_locations_ext.get_data()
nwb_unit_locations = UnitLocations(
    name="unit_locations",
    data=unit_locations_data,
)

# ---- Step 7: Assemble the SpikeSortingContainer and write to NWB ----

sparsity_mask = sparsity.mask if sparsity is not None else None

extensions = SpikeSortingExtensions(name="extensions")
extensions.random_spikes = nwb_random_spikes
extensions.templates = nwb_templates
extensions.noise_levels = nwb_noise_levels
extensions.unit_locations = nwb_unit_locations

container = SpikeSortingContainer(
    name="spike_sorting",
    sampling_frequency=recording.get_sampling_frequency(),
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

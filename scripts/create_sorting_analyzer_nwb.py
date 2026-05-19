"""
Script to create a SortingAnalyzer from SpikeInterface and save it to NWB.

This script demonstrates how to:
1. Generate mock recording and sorting data using SpikeInterface
2. Create a SortingAnalyzer and compute extensions (random_spikes, templates)
3. Write recording and sorting to NWB via neuroconv
4. Add ndx-spikesorting extension data via add_sorting_analyzer_to_nwbfile
"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from neuroconv.tools.spikeinterface import add_recording_to_nwbfile, add_sorting_to_nwbfile
from pynwb import NWBHDF5IO, NWBFile
from spikeinterface.core.base import unit_period_dtype
from spikeinterface import create_sorting_analyzer, generate_ground_truth_recording, set_global_job_kwargs

from ndx_spikesorting.utils import add_sorting_analyzer_to_nwbfile

set_global_job_kwargs(n_jobs=-1)

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
        "spike_locations": {"method": "grid_convolution"},
        # Metrics: SI's quality_metrics, template_metrics, spiketrain_metrics
        # get redistributed by the writer. Cell-intrinsic metrics
        # (firing_rate from spiketrain, peak_to_valley and half_width from
        # template) land as canonical typed columns on nwbfile.units.
        # Run-dependent metrics (snr, presence_ratio, isi_violations_ratio,
        # amplitude_cutoff, amplitude_median from quality) land as canonical
        # typed columns on a MetricsRun instance inside SpikeSortingExtensions.
        "quality_metrics": {},
        "template_metrics": {},
        "spiketrain_metrics": {},
    }
)

# Create user-defined valid periods for each unit
# Each unit gets two disjoint valid periods
num_units = len(sorting.unit_ids)
sampling_frequency = recording.sampling_frequency
user_defined_periods = np.array([], dtype=unit_period_dtype)
for unit_index in range(num_units):
    # First valid period: 0.5s–2s
    user_defined_periods = np.append(
        user_defined_periods,
        np.array(
            [(0, int(0.5 * sampling_frequency), int(2.0 * sampling_frequency), unit_index)],
            dtype=unit_period_dtype)
        )
    # Second valid period: 3s–4.5s
    user_defined_periods = np.append(
        user_defined_periods,
        np.array(
            [(0, int(3.0 * sampling_frequency), int(4.5 * sampling_frequency), unit_index)],
            dtype=unit_period_dtype
        )
    )

sorting_analyzer.compute(
    "valid_unit_periods",
    method="user_defined",
    user_defined_periods=user_defined_periods,
    minimum_valid_period_duration=0,
)

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

# ---- Step 3: Add sorting analyzer extensions to NWB ----

add_sorting_analyzer_to_nwbfile(sorting_analyzer=sorting_analyzer, nwbfile=nwbfile)

# ---- Step 4: Write to disk ----

output_file = Path(__file__).parent / "sorting_analyzer_test.nwb"
with NWBHDF5IO(str(output_file), mode="w") as io:
    io.write(nwbfile)

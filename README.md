# ndx-spikesorting Extension for NWB

`ndx-spikesorting` is an NWB extension for storing spike sorting outputs and the derived
quantities ("extensions") commonly produced by spike sorting analyses, such as templates,
waveforms, quality metrics, and PCA projections. The data model closely mirrors the
[SpikeInterface](https://github.com/SpikeInterface/spikeinterface) `SortingAnalyzer` so that
the round trip between an in-memory analyzer and an NWB file is lossless for the supported
extensions.

The extension introduces the following new neurodata types:

1. A `SpikeSortingContainer` type (extends `NWBDataInterface`) that aggregates the results of
   a single sorting analysis. It stores the sampling frequency, a `DynamicTableRegion`
   reference to the rows of the electrodes table used for sorting, a `DynamicTableRegion`
   reference to the rows of the units table, an optional `(num_units, num_channels)` boolean
   `sparsity_mask`, an optional link to the source `ElectricalSeries`, and a child
   `SpikeSortingExtensions` group with the computed extensions.
2. A `SpikeSortingExtensions` type (extends `NWBDataInterface`) that holds the optional
   computed extension objects listed below. Every extension is optional, so users only store
   what has actually been computed.
3. A `RandomSpikes` type (extends `NWBDataInterface`) that stores, as a ragged array, the
   randomly selected spike indices per unit used for waveform extraction and template
   computation.
4. A `Waveforms` type (extends `NWBDataInterface`) that stores individual spike waveforms as
   a *double-ragged* array. The first `VectorIndex` groups channel-waveform rows by spike;
   the second groups spikes by unit. A `DynamicTableRegion` named `electrodes` records which
   electrode each row corresponds to. The `peak_sample_index` attribute records the
   alignment point.
5. A `Templates` type (extends `NWBDataInterface`) that stores template waveforms per unit as
   a ragged array. When sparsity is used, only active channels are stored per unit. A
   `DynamicTableRegion` named `electrodes` records the electrode for each row. The
   `peak_sample_index` attribute records the alignment point.
6. A `NoiseLevels` type (extends `NWBDataInterface`) that stores estimated noise levels per
   channel (length matches `SpikeSortingContainer.electrodes`).
7. A `UnitLocations` type (extends `NWBDataInterface`) that stores 2D or 3D estimated
   locations per unit.
8. A `Correlograms` type (extends `NWBDataInterface`) that stores per-unit-pair correlogram
   spike counts together with the bin edges in milliseconds.
9. An `ISIHistograms` type (extends `NWBDataInterface`) that stores per-unit inter-spike
   interval histograms with their bin edges in milliseconds.
10. A `TemplateSimilarity` type (extends `NWBDataInterface`) that stores a
    `(num_units, num_units)` similarity matrix between unit templates.
11. A `SpikeAmplitudes` type (extends `NWBDataInterface`) that stores per-spike amplitudes
    as a ragged array indexed per unit.
12. A `SpikeLocations` type (extends `NWBDataInterface`) that stores per-spike 2D or 3D
    estimated locations as a ragged array indexed per unit.
13. An `AmplitudeScalings` type (extends `NWBDataInterface`) that stores the amplitude
    scaling of each spike relative to its template, as a ragged array indexed per unit.
14. A `PCAProjectionsByChannel` type (extends `NWBDataInterface`) that stores per-channel
    PCA projections of spikes as a double-ragged array (rows by spike, spikes by unit),
    along with the corresponding `electrodes` table region and a link back to the
    `Waveforms` from which the projections were computed.
15. A `PCAProjectionsConcatenated` type (extends `NWBDataInterface`) that stores
    "concatenated-channels" PCA projections as a single-ragged array, with a link back to
    the `Waveforms` from which the projections were computed.
16. A `ValidUnitPeriods` type (extends `TimeIntervals`) that stores valid time periods per
    unit (e.g. derived from drift, false-positive/false-negative estimates, or user
    curation). Each row references one unit via a `DynamicTableRegion`; units with multiple
    disjoint valid periods are encoded as multiple rows referencing the same unit.

## Installation

```bash
pip install ndx-spikesorting
```

## Usage

Round-trip a SpikeInterface `SortingAnalyzer` through an NWB file and open the result in
[spikeinterface-gui](https://github.com/SpikeInterface/spikeinterface-gui):

```python
from datetime import datetime, timezone

from pynwb import NWBFile, NWBHDF5IO
from neuroconv.tools.spikeinterface import add_recording_to_nwbfile, add_sorting_to_nwbfile
from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording
from spikeinterface_gui import run_mainwindow

from ndx_spikesorting import add_sorting_analyzer_to_nwbfile, read_sorting_analyzer_from_nwb

# 1) Build a SortingAnalyzer with some computed extensions
recording, sorting = generate_ground_truth_recording(
    durations=[5.0], num_units=5, num_channels=10, seed=42
)
sorting_analyzer = create_sorting_analyzer(
    sorting=sorting, recording=recording, format="memory", sparse=True
)
sorting_analyzer.compute([
    "random_spikes", "waveforms", "templates", "noise_levels",
    "unit_locations", "correlograms", "isi_histograms", "template_similarity",
    "spike_amplitudes", "spike_locations", "amplitude_scalings", "principal_components",
])

# 2) Write the recording, units, and SortingAnalyzer extensions to an NWB file
nwbfile = NWBFile(
    session_description="Demo",
    identifier="demo",
    session_start_time=datetime.now(timezone.utc),
)
add_recording_to_nwbfile(recording, nwbfile=nwbfile, write_as="raw", iterator_type=None)
add_sorting_to_nwbfile(sorting, nwbfile=nwbfile, write_as="units")
add_sorting_analyzer_to_nwbfile(sorting_analyzer, nwbfile=nwbfile)

with NWBHDF5IO("demo_sorting.nwb", mode="w") as io:
    io.write(nwbfile)

# 3) Read the file back into a SortingAnalyzer and launch the GUI
reloaded_analyzer = read_sorting_analyzer_from_nwb("demo_sorting.nwb")
run_mainwindow(reloaded_analyzer, mode="desktop")
```

## Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#ffffff', 'primaryBorderColor': '#144E73', 'lineColor': '#D96F32'}}}%%

classDiagram
    direction LR

    class SpikeSortingContainer {
        <<NWBDataInterface>>

        sampling_frequency : float
        sparsity_mask : NDArray[Shape["num_units, num_channels"], Bool], optional
        electrodes : DynamicTableRegion
        units_region : DynamicTableRegion
        source_electrical_series : ElectricalSeries, optional
        spike_sorting_extensions : SpikeSortingExtensions, optional
    }

    class SpikeSortingExtensions {
        <<NWBDataInterface>>

        random_spikes : RandomSpikes, optional
        waveforms : Waveforms, optional
        templates : Templates, optional
        noise_levels : NoiseLevels, optional
        unit_locations : UnitLocations, optional
        correlograms : Correlograms, optional
        isi_histograms : ISIHistograms, optional
        template_similarity : TemplateSimilarity, optional
        spike_amplitudes : SpikeAmplitudes, optional
        spike_locations : SpikeLocations, optional
        amplitude_scalings : AmplitudeScalings, optional
        pca_projections_by_channel : PCAProjectionsByChannel, optional
        pca_projections_concatenated : PCAProjectionsConcatenated, optional
        valid_unit_periods : ValidUnitPeriods, optional
    }

    class RandomSpikes {
        <<NWBDataInterface>>

        random_spikes_indices : VectorData[NDArray[Shape["*"], Int64]]
        random_spikes_indices_index : VectorIndex
    }

    class Waveforms {
        <<NWBDataInterface>>

        peak_sample_index : int32
        data : VectorData[NDArray[Shape["*, *"], Float32]]
        data_index : VectorIndex
        data_index_index : VectorIndex
        electrodes : DynamicTableRegion
        random_spikes : RandomSpikes (link)
    }

    class Templates {
        <<NWBDataInterface>>

        peak_sample_index : int32
        data : VectorData[NDArray[Shape["*, *"], Float32]]
        data_index : VectorIndex
        electrodes : DynamicTableRegion
    }

    class NoiseLevels {
        <<NWBDataInterface>>

        data : NDArray[Shape["num_channels"], Float32]
        --> unit : str = "microvolts"
    }

    class UnitLocations {
        <<NWBDataInterface>>

        data : NDArray[Shape["num_units, {2|3}"], Float]
        --> unit : str = "micrometers"
    }

    class Correlograms {
        <<NWBDataInterface>>

        data : NDArray[Shape["num_units, num_units, num_bins"], Int]
        bin_edges : NDArray[Shape["num_bin_edges"], Float]
    }

    class ISIHistograms {
        <<NWBDataInterface>>

        data : NDArray[Shape["num_units, num_bins"], Int]
        bin_edges : NDArray[Shape["num_bin_edges"], Float]
    }

    class TemplateSimilarity {
        <<NWBDataInterface>>

        data : NDArray[Shape["num_units, num_units"], Float]
    }

    class SpikeAmplitudes {
        <<NWBDataInterface>>

        data : VectorData[NDArray[Shape["*"], Float]]
        data_index : VectorIndex
    }

    class SpikeLocations {
        <<NWBDataInterface>>

        data : VectorData[NDArray[Shape["*, {2|3}"], Float]]
        data_index : VectorIndex
    }

    class AmplitudeScalings {
        <<NWBDataInterface>>

        data : VectorData[NDArray[Shape["*"], Float32]]
        data_index : VectorIndex
    }

    class PCAProjectionsByChannel {
        <<NWBDataInterface>>

        data : VectorData[NDArray[Shape["*, num_components"], Float]]
        data_index : VectorIndex
        data_index_index : VectorIndex
        electrodes : DynamicTableRegion
        waveforms : Waveforms (link)
    }

    class PCAProjectionsConcatenated {
        <<NWBDataInterface>>

        data : VectorData[NDArray[Shape["*, num_components"], Float]]
        data_index : VectorIndex
        waveforms : Waveforms (link)
    }

    class ValidUnitPeriods {
        <<TimeIntervals>>

        start_time : VectorData[NDArray[Shape["*"], Float]]
        stop_time : VectorData[NDArray[Shape["*"], Float]]
        unit : DynamicTableRegion
    }

    SpikeSortingContainer "1" *--> "0..1" SpikeSortingExtensions : contains
    SpikeSortingExtensions "1" *--> "0..1" RandomSpikes : contains
    SpikeSortingExtensions "1" *--> "0..1" Waveforms : contains
    SpikeSortingExtensions "1" *--> "0..1" Templates : contains
    SpikeSortingExtensions "1" *--> "0..1" NoiseLevels : contains
    SpikeSortingExtensions "1" *--> "0..1" UnitLocations : contains
    SpikeSortingExtensions "1" *--> "0..1" Correlograms : contains
    SpikeSortingExtensions "1" *--> "0..1" ISIHistograms : contains
    SpikeSortingExtensions "1" *--> "0..1" TemplateSimilarity : contains
    SpikeSortingExtensions "1" *--> "0..1" SpikeAmplitudes : contains
    SpikeSortingExtensions "1" *--> "0..1" SpikeLocations : contains
    SpikeSortingExtensions "1" *--> "0..1" AmplitudeScalings : contains
    SpikeSortingExtensions "1" *--> "0..1" PCAProjectionsByChannel : contains
    SpikeSortingExtensions "1" *--> "0..1" PCAProjectionsConcatenated : contains
    SpikeSortingExtensions "1" *--> "0..1" ValidUnitPeriods : contains
    Waveforms ..> RandomSpikes : link
    PCAProjectionsByChannel ..> Waveforms : link
    PCAProjectionsConcatenated ..> Waveforms : link
```

## Developer installation

In a Python 3.10-3.13 environment:
```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

Install pre-commit hooks:
```bash
pre-commit install
```

Style and other checks:
```bash
black .
ruff .
codespell .
```

---
This extension was created using [ndx-template](https://github.com/nwb-extensions/ndx-template).

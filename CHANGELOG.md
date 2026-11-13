# Changelog for ndx-spikesorting

# v0.1.2 (Upcoming)

<<<<<<< HEAD
### Fixes

* `read_sorting_analyzer_from_nwb` reads a container that links no `ElectricalSeries` (e.g. a file with only the sorting, or only LFP) as a recordingless `SortingAnalyzer`. Its channel ids, probe and sparsity come from the electrodes table and the container. The waveforms extension also gets its `ms_before`/`ms_after` from the stored peak index instead of the defaults.
* `add_sorting_analyzer_to_nwbfile` links an `ElectricalSeries` only when `electrical_series_name` is given, and never picks one from the file. `electrical_series_name` is now the series' path in the file (e.g. `"acquisition/ElectricalSeries"` or `"processing/ecephys/LFP/ElectricalSeriesLFP"`); a bare name is still looked up in acquisition. It checks that the series covers the analyzer's electrodes and is sampled at its rate, which rules out e.g. an LFP series.
=======
### Features

* `SpikeSortingContainer` has an optional `num_samples` attribute, the number of samples of the recording the analysis was computed on. `add_sorting_analyzer_to_nwbfile` writes it, so the recording duration is stored even when no `ElectricalSeries` is linked.
>>>>>>> 8a3c41cfe2079e395a5901d54a860f00c581c6c5

# v0.1.1

**Released: Nov 10th 2026**

### Fixes

* `add_sorting_analyzer_to_nwbfile` no longer assumes the analyzer owns the electrodes and units tables. The container's `electrodes` region and the `electrodes` regions of `Templates`, `Waveforms` and `PCAProjectionsByChannel` now store the electrodes-table rows of the analyzer's channels (matched by `channel_name`, or given with the new `electrode_indices`), and `units_region` stores the rows of its units (matched by `unit_name`). This makes files from multi-probe sessions, or with units from several sortings, read back correctly. `read_sorting_analyzer_from_nwb` reads only the units and channels the container references, so files written before this change read the same way.
* `add_sorting_analyzer_to_nwbfile` takes a `units_table` object instead of `unit_table_name`, which looked the name up as a processing module. All metrics now go to that table, so a units table in processing works.
* Added `container_name` to `add_sorting_analyzer_to_nwbfile`, so several analyzers can be written to one file. The `firing_rate`/`num_spikes` typed columns are only added to the units table if missing; otherwise the analyzer's values stay in its `UnitsMetrics`.

### Features

* Added `Waveforms` and `PrincipalComponents` extensions
* Added `BaseVectorExtensions` (spike amplitudes, locations, scalings)
* Added `NoiseLevels` extension
* Added `UnitLocations` extension
* Added `templates_to_dense` method to the `utils` class for converting sparse ragged templates to dense 3D arrays. [PR #3](https://github.com/catalystneuro/ndx-spikesorting/pull/3)
* Added `read_sorting_analyzer_from_nwb` loader script for reconstructing a SpikeInterface `SortingAnalyzer` from an ndx-spikesorting NWB file with precomputed extensions (random_spikes, templates). [PR #3](https://github.com/catalystneuro/ndx-spikesorting/pull/3)

### Dependencies

* Depends on `neuroconv>=0.10` and support for Python 3.11-3.13


# v0.1.0

**Released: Jun 10th 2026**

First release
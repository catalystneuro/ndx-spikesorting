# Changelog for ndx-spikesorting

# v0.1.1 (Upcoming)

### Fixes

* `add_sorting_analyzer_to_nwbfile` no longer assumes the analyzer owns the electrodes and units tables. The container's `electrodes` region and the `electrodes` regions of `Templates`, `Waveforms` and `PCAProjectionsByChannel` now store the electrodes-table rows of the analyzer's channels (matched by `channel_name`, or given with the new `electrode_indices`), and `units_region` stores the rows of its units (matched by `unit_name`). This makes files from multi-probe sessions, or with units from several sortings, read back correctly. `read_sorting_analyzer_from_nwb` reads only the units and channels the container references, so files written before this change read the same way.
* `add_sorting_analyzer_to_nwbfile` takes a `units_table` object instead of `unit_table_name`, which looked the name up as a processing module. All metrics now go to that table, so a units table in processing works.
* With an explicit `electrical_series_name`, `add_sorting_analyzer_to_nwbfile` checks that the series covers the analyzer's electrodes. The error for several series and no name lists each candidate's electrodes rows.
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
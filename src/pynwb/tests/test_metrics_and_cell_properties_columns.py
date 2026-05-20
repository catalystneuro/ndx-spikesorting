"""Round-trip tests for canonical typed VectorData columns on nwbfile.units.

Cell-intrinsic typed columns (FiringRate, NumSpikes, ...) live on
``nwbfile.units`` rather than inside the SpikeSortingExtensions container.
This module exercises their writer + reader paths and the SpikeInterface
extension-aliasing policy committed by ``UNITS_TYPED_COLUMNS``.
"""

import numpy as np

from pynwb import NWBHDF5IO

from ndx_spikesorting import (
    FiringRate,
    NumSpikes,
    add_sorting_analyzer_to_nwbfile,
    read_sorting_analyzer_from_nwb,
)


def test_firing_rate_column_is_typed(sorting_analyzer_with_extensions, nwbfile_with_recording_and_sorting):
    """firing_rate column on nwbfile.units is a FiringRate instance."""
    sa, _, _ = sorting_analyzer_with_extensions
    add_sorting_analyzer_to_nwbfile(sa, nwbfile_with_recording_and_sorting)

    assert "firing_rate" in nwbfile_with_recording_and_sorting.units.colnames
    assert isinstance(nwbfile_with_recording_and_sorting.units["firing_rate"], FiringRate)


def test_num_spikes_column_is_typed(sorting_analyzer_with_extensions, nwbfile_with_recording_and_sorting):
    """num_spikes column on nwbfile.units is a NumSpikes instance."""
    sa, _, _ = sorting_analyzer_with_extensions
    add_sorting_analyzer_to_nwbfile(sa, nwbfile_with_recording_and_sorting)

    assert "num_spikes" in nwbfile_with_recording_and_sorting.units.colnames
    assert isinstance(nwbfile_with_recording_and_sorting.units["num_spikes"], NumSpikes)


def test_values_match_source(sorting_analyzer_with_extensions, nwbfile_with_recording_and_sorting):
    """Typed columns carry the values from the SI extension."""
    sa, _, _ = sorting_analyzer_with_extensions
    add_sorting_analyzer_to_nwbfile(sa, nwbfile_with_recording_and_sorting)

    df = sa.get_extension("quality_metrics").get_data()

    np.testing.assert_array_almost_equal(
        np.asarray(nwbfile_with_recording_and_sorting.units["firing_rate"].data[:]),
        df["firing_rate"].to_numpy(),
        decimal=5,
    )
    np.testing.assert_array_equal(
        np.asarray(nwbfile_with_recording_and_sorting.units["num_spikes"].data[:]),
        df["num_spikes"].to_numpy(),
    )


def test_round_trip_to_metric_extensions(
    sorting_analyzer_with_extensions, nwbfile_with_recording_and_sorting, tmp_path
):
    """After write/read, typed columns reconstruct their SI metric extensions.

    ``firing_rate`` and ``num_spikes`` are produced by both ``quality_metrics``
    and ``spiketrain_metrics``, so on read they are routed back into every SI
    extension that defines them (route-to-both, via
    ``find_extension_metric_targets``).
    """
    sa, _, _ = sorting_analyzer_with_extensions
    add_sorting_analyzer_to_nwbfile(sa, nwbfile_with_recording_and_sorting)

    path = tmp_path / "test_cell_intrinsic_columns.nwb"
    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile_with_recording_and_sorting)

    sa_restored = read_sorting_analyzer_from_nwb(path)

    original_df = sa.get_extension("quality_metrics").get_data()
    for ext_name in ("quality_metrics", "spiketrain_metrics"):
        assert ext_name in sa_restored.extensions
        df = sa_restored.get_extension(ext_name).get_data()
        assert "firing_rate" in df.columns
        assert "num_spikes" in df.columns
        np.testing.assert_array_almost_equal(
            df["firing_rate"].to_numpy(),
            original_df["firing_rate"].to_numpy(),
            decimal=5,
        )
        np.testing.assert_array_equal(
            df["num_spikes"].to_numpy(),
            original_df["num_spikes"].to_numpy(),
        )

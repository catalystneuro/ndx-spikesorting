"""Round trips where the analyzer shares the electrodes, units or acquisition with other data.

Each test builds a file in which the analyzer's channels or units are not rows ``0..n-1``
of their tables (other probes, other sortings, other recordings), as neuroconv produces
for multi-probe sessions, and checks that every region points at the analyzer's own rows.
"""

from datetime import datetime, timezone

import numpy as np
import pytest
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries

from ndx_spikesorting import add_sorting_analyzer_to_nwbfile, read_sorting_analyzer_from_nwb

NUM_CHANNELS = 10
NUM_UNITS = 5


@pytest.fixture(scope="module")
def analyzer():
    from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording

    recording, sorting = generate_ground_truth_recording(
        durations=[5.0], num_units=NUM_UNITS, num_channels=NUM_CHANNELS, seed=0
    )
    sorting_analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format="memory", sparse=True)
    sorting_analyzer.compute(
        {
            "random_spikes": {"max_spikes_per_unit": 10, "seed": 0},
            "waveforms": {},
            "templates": {},
            "noise_levels": {},
            "principal_components": {"n_components": 3, "mode": "by_channel_local"},
            "quality_metrics": {"metric_names": ["firing_rate", "num_spikes", "snr"]},
        }
    )
    return sorting_analyzer


def _new_nwbfile():
    return NWBFile(session_description="test", identifier="test", session_start_time=datetime.now(timezone.utc))


def _other_probe(recording, rename=True):
    """A recording on another probe, named 'other', with renamed channels unless `rename` is False."""
    other = recording.rename_channels([f"other_{c}" for c in recording.channel_ids]) if rename else recording.clone()
    other.set_property("group_name", ["other"] * other.get_num_channels())
    return other


def _write_and_read(nwbfile, path, container_path="ecephys/spike_sorting"):
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    return read_sorting_analyzer_from_nwb(path, container_path=container_path)


def _assert_same_analyzer(restored, original):
    assert [str(u) for u in restored.unit_ids] == [str(u) for u in original.unit_ids]
    np.testing.assert_array_equal(restored.sparsity.mask, original.sparsity.mask)
    np.testing.assert_array_almost_equal(
        restored.get_extension("templates").data["average"],
        original.get_extension("templates").data["average"],
        decimal=5,
    )
    restored_qm = restored.get_extension("quality_metrics").get_data()
    original_qm = original.get_extension("quality_metrics").get_data()
    for column in ("firing_rate", "num_spikes", "snr"):
        np.testing.assert_allclose(restored_qm[column].to_numpy(float), original_qm[column].to_numpy(float))


# ---------------------------------------------------------------------------
# 1. Electrodes table with rows from another probe
# ---------------------------------------------------------------------------


def test_electrodes_after_another_probe_round_trip(analyzer, tmp_path):
    from neuroconv.tools.spikeinterface import (
        add_recording_metadata_to_nwbfile,
        add_recording_to_nwbfile,
        add_sorting_to_nwbfile,
    )

    nwbfile = _new_nwbfile()
    add_recording_metadata_to_nwbfile(_other_probe(analyzer.recording), nwbfile=nwbfile)
    add_recording_to_nwbfile(analyzer.recording, nwbfile=nwbfile, iterator_type=None)
    add_sorting_to_nwbfile(analyzer.sorting, nwbfile=nwbfile)
    assert len(nwbfile.electrodes) == 2 * NUM_CHANNELS

    add_sorting_analyzer_to_nwbfile(analyzer, nwbfile)

    container = nwbfile.processing["ecephys"]["spike_sorting"]
    analyzer_rows = list(range(NUM_CHANNELS, 2 * NUM_CHANNELS))
    assert list(container.electrodes.data) == analyzer_rows
    extensions = container.spike_sorting_extensions
    for nwb_object in (extensions.templates, extensions.waveforms, extensions.pca_projections_by_channel):
        assert set(nwb_object.electrodes.data) <= set(analyzer_rows)

    _assert_same_analyzer(_write_and_read(nwbfile, tmp_path / "file.nwb"), analyzer)


def test_ambiguous_channel_names_need_electrode_indices(analyzer, tmp_path):
    from neuroconv.tools.spikeinterface import (
        add_recording_metadata_to_nwbfile,
        add_recording_to_nwbfile,
        add_sorting_to_nwbfile,
    )

    nwbfile = _new_nwbfile()
    add_recording_metadata_to_nwbfile(_other_probe(analyzer.recording, rename=False), nwbfile=nwbfile)
    add_recording_to_nwbfile(analyzer.recording, nwbfile=nwbfile, iterator_type=None)
    add_sorting_to_nwbfile(analyzer.sorting, nwbfile=nwbfile)

    with pytest.raises(ValueError, match="appear more than once"):
        add_sorting_analyzer_to_nwbfile(analyzer, nwbfile)

    analyzer_rows = list(range(NUM_CHANNELS, 2 * NUM_CHANNELS))
    add_sorting_analyzer_to_nwbfile(analyzer, nwbfile, electrode_indices=analyzer_rows)
    assert list(nwbfile.processing["ecephys"]["spike_sorting"].electrodes.data) == analyzer_rows
    _assert_same_analyzer(_write_and_read(nwbfile, tmp_path / "file.nwb"), analyzer)


# ---------------------------------------------------------------------------
# 2. Units table with rows from another sorting, or in processing
# ---------------------------------------------------------------------------


def test_units_after_another_sorting_round_trip(analyzer, tmp_path):
    from neuroconv.tools.spikeinterface import add_recording_to_nwbfile, add_sorting_to_nwbfile

    nwbfile = _new_nwbfile()
    add_recording_to_nwbfile(analyzer.recording, nwbfile=nwbfile, iterator_type=None)
    other_sorting = analyzer.sorting.rename_units([f"other_{u}" for u in analyzer.unit_ids])
    add_sorting_to_nwbfile(other_sorting, nwbfile=nwbfile)
    add_sorting_to_nwbfile(analyzer.sorting, nwbfile=nwbfile)
    assert len(nwbfile.units) == 2 * NUM_UNITS

    add_sorting_analyzer_to_nwbfile(analyzer, nwbfile)

    container = nwbfile.processing["ecephys"]["spike_sorting"]
    assert list(container.units_region.data) == list(range(NUM_UNITS, 2 * NUM_UNITS))
    firing_rate = np.asarray(nwbfile.units["firing_rate"].data)
    assert np.isnan(firing_rate[:NUM_UNITS]).all()
    assert list(nwbfile.units["num_spikes"].data[:NUM_UNITS]) == [-1] * NUM_UNITS

    _assert_same_analyzer(_write_and_read(nwbfile, tmp_path / "file.nwb"), analyzer)


def test_units_table_in_processing_round_trip(analyzer, tmp_path):
    from neuroconv.tools.spikeinterface import add_recording_to_nwbfile, add_sorting_to_nwbfile

    nwbfile = _new_nwbfile()
    add_recording_to_nwbfile(analyzer.recording, nwbfile=nwbfile, iterator_type=None)
    add_sorting_to_nwbfile(analyzer.sorting, nwbfile=nwbfile, write_as="processing", units_name="sorted_units")
    units_table = nwbfile.processing["ecephys"]["sorted_units"]
    assert nwbfile.units is None

    add_sorting_analyzer_to_nwbfile(analyzer, nwbfile, units_table=units_table)

    assert nwbfile.processing["ecephys"]["spike_sorting"].units_region.table is units_table
    assert "firing_rate" in units_table.colnames
    _assert_same_analyzer(_write_and_read(nwbfile, tmp_path / "file.nwb"), analyzer)


# ---------------------------------------------------------------------------
# 3. Several ElectricalSeries in acquisition
# ---------------------------------------------------------------------------


def _file_with_two_probes(analyzer):
    """Analyzer channels are electrodes rows 10..19, recorded by the series neuroconv adds.

    Also adds 'OtherSeries' over the other probe's rows 0..9 and 'AllSeries' over all 20 rows.
    Returns the file and the name of the analyzer's own series.
    """
    from neuroconv.tools.spikeinterface import (
        add_recording_metadata_to_nwbfile,
        add_recording_to_nwbfile,
        add_sorting_to_nwbfile,
    )

    nwbfile = _new_nwbfile()
    add_recording_metadata_to_nwbfile(_other_probe(analyzer.recording), nwbfile=nwbfile)
    add_recording_to_nwbfile(analyzer.recording, nwbfile=nwbfile, iterator_type=None)
    add_sorting_to_nwbfile(analyzer.sorting, nwbfile=nwbfile)
    (analyzer_series_name,) = nwbfile.acquisition
    for name, rows in (("OtherSeries", range(NUM_CHANNELS)), ("AllSeries", range(2 * NUM_CHANNELS))):
        nwbfile.add_acquisition(
            ElectricalSeries(
                name=name,
                data=np.zeros((analyzer.get_num_samples(), len(rows)), dtype="int16"),
                electrodes=nwbfile.create_electrode_table_region(region=list(rows), description=name),
                rate=analyzer.sampling_frequency,
            )
        )
    return nwbfile, analyzer_series_name


def test_several_electrical_series_require_a_name(analyzer):
    nwbfile, _ = _file_with_two_probes(analyzer)
    with pytest.raises(ValueError, match="Specify electrical_series_name explicitly"):
        add_sorting_analyzer_to_nwbfile(analyzer, nwbfile)


def test_electrical_series_name_links_that_series(analyzer, tmp_path):
    nwbfile, series_name = _file_with_two_probes(analyzer)
    add_sorting_analyzer_to_nwbfile(analyzer, nwbfile, electrical_series_name=series_name)

    container = nwbfile.processing["ecephys"]["spike_sorting"]
    assert container.source_electrical_series is nwbfile.acquisition[series_name]
    restored = _write_and_read(nwbfile, tmp_path / "file.nwb")
    assert restored.has_recording()
    _assert_same_analyzer(restored, analyzer)


def test_electrical_series_not_covering_the_analyzer_raises(analyzer):
    nwbfile, _ = _file_with_two_probes(analyzer)
    with pytest.raises(ValueError, match="does not cover"):
        add_sorting_analyzer_to_nwbfile(analyzer, nwbfile, electrical_series_name="OtherSeries")


def test_electrical_series_covering_more_channels_is_restricted_on_read(analyzer, tmp_path):
    nwbfile, _ = _file_with_two_probes(analyzer)
    add_sorting_analyzer_to_nwbfile(analyzer, nwbfile, electrical_series_name="AllSeries")

    restored = _write_and_read(nwbfile, tmp_path / "file.nwb")
    assert restored.get_num_channels() == NUM_CHANNELS
    assert [str(c) for c in restored.channel_ids] == [str(c) for c in analyzer.channel_ids]
    _assert_same_analyzer(restored, analyzer)


# ---------------------------------------------------------------------------
# 4. Two analyzers in one file
# ---------------------------------------------------------------------------


def test_two_analyzers_share_the_units_table(analyzer, tmp_path):
    from neuroconv.tools.spikeinterface import add_recording_to_nwbfile, add_sorting_to_nwbfile

    nwbfile = _new_nwbfile()
    add_recording_to_nwbfile(analyzer.recording, nwbfile=nwbfile, iterator_type=None)
    add_sorting_to_nwbfile(analyzer.sorting, nwbfile=nwbfile)
    subset = analyzer.select_units(analyzer.unit_ids[[3, 1]])

    add_sorting_analyzer_to_nwbfile(analyzer, nwbfile)
    with pytest.raises(ValueError, match="already holds an object named 'spike_sorting'"):
        add_sorting_analyzer_to_nwbfile(subset, nwbfile)
    add_sorting_analyzer_to_nwbfile(subset, nwbfile, container_name="spike_sorting_subset")

    subset_container = nwbfile.processing["ecephys"]["spike_sorting_subset"]
    assert list(subset_container.units_region.data) == [3, 1]
    # The typed firing_rate column belongs to the first analyzer, so the subset keeps its own in UnitsMetrics.
    subset_metrics = subset_container.spike_sorting_extensions.units_metrics["unit_metrics"]
    assert "firing_rate" in subset_metrics.colnames

    path = tmp_path / "file.nwb"
    _assert_same_analyzer(_write_and_read(nwbfile, path), analyzer)
    _assert_same_analyzer(read_sorting_analyzer_from_nwb(path, container_path="ecephys/spike_sorting_subset"), subset)

"""Reading a container that links no ElectricalSeries returns a recordingless SortingAnalyzer."""

from datetime import datetime, timezone

import numpy as np
from pynwb import NWBHDF5IO, NWBFile

from ndx_spikesorting import add_sorting_analyzer_to_nwbfile, read_sorting_analyzer_from_nwb


def test_round_trip_without_electrical_series(tmp_path):
    from neuroconv.tools.spikeinterface import add_recording_metadata_to_nwbfile, add_sorting_to_nwbfile
    from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording

    recording, sorting = generate_ground_truth_recording(durations=[3.0], num_units=4, num_channels=6, seed=0)
    for key in ("gt_unit_locations", "main_channel_id"):
        if key in sorting.get_property_keys():
            sorting.delete_property(key)
    for key, column in zip(("rel_x", "rel_y"), recording.get_channel_locations().T):
        recording.set_property(key, column)
    analyzer = create_sorting_analyzer(sorting=sorting, recording=recording, format="memory", sparse=True)
    analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels", "principal_components"])

    # Electrodes and units only: no ElectricalSeries, so the container links none.
    nwbfile = NWBFile(session_description="test", identifier="test", session_start_time=datetime.now(timezone.utc))
    add_recording_metadata_to_nwbfile(recording, nwbfile=nwbfile)
    add_sorting_to_nwbfile(sorting, nwbfile=nwbfile)
    add_sorting_analyzer_to_nwbfile(analyzer, nwbfile)
    path = tmp_path / "recordingless.nwb"
    with NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)

    restored = read_sorting_analyzer_from_nwb(path)

    assert not restored.has_recording()
    assert [str(c) for c in restored.channel_ids] == [str(c) for c in analyzer.channel_ids]
    np.testing.assert_allclose(restored.get_channel_locations(), analyzer.get_channel_locations())
    np.testing.assert_array_equal(restored.sparsity.mask, analyzer.sparsity.mask)
    np.testing.assert_array_almost_equal(
        restored.get_extension("templates").data["average"],
        analyzer.get_extension("templates").data["average"],
        decimal=5,
    )
    waveforms = restored.get_extension("waveforms")
    assert waveforms.nbefore == analyzer.get_extension("waveforms").nbefore
    unit_id = restored.unit_ids[0]
    assert waveforms.get_waveforms_one_unit(unit_id, force_dense=False).shape == analyzer.get_extension(
        "waveforms"
    ).get_waveforms_one_unit(analyzer.unit_ids[0], force_dense=False).shape
    # spikeinterface>=0.106 resolves main channels from the stored templates when the analyzer has no recording.
    if hasattr(restored, "main_channel_indices"):
        np.testing.assert_array_equal(restored.main_channel_indices, analyzer.main_channel_indices)

    # Extensions that need no traces can still be computed.
    restored.compute(["unit_locations", "template_similarity"])
    analyzer.compute(["unit_locations", "template_similarity"])
    np.testing.assert_allclose(
        restored.get_extension("template_similarity").get_data(),
        analyzer.get_extension("template_similarity").get_data(),
        atol=1e-4,
    )

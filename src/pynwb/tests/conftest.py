"""Shared pytest fixtures for the ndx-spikesorting test suite.

These fixtures are pytest-only; ``TestCase`` tests in ``test_spikesorting.py``
keep their local module-level helpers until they migrate to pytest.
"""

import pytest


@pytest.fixture(scope="module")
def sorting_analyzer_with_extensions():
    """SortingAnalyzer with all standard SI extensions and metrics computed.

    Module-scoped so the compute step runs once per test file rather than per
    test. Tests should not mutate the returned objects. Add new SI extensions
    here as future canonization PRs need them (template_metrics, etc.).
    """
    from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording

    recording, sorting = generate_ground_truth_recording(
        durations=[5.0],
        num_units=5,
        num_channels=10,
        seed=42,
    )
    sa = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="memory",
        sparse=True,
    )
    sa.compute(
        {
            "random_spikes": {"max_spikes_per_unit": 10, "seed": 42},
            "waveforms": {},
            "templates": {},
            "noise_levels": {},
            "spike_amplitudes": {},
            "quality_metrics": {"metric_names": ["firing_rate", "num_spikes"]},
        }
    )
    return sa, recording, sorting


@pytest.fixture
def nwbfile_with_recording_and_sorting(sorting_analyzer_with_extensions):
    """Fresh NWBFile populated with the module-scoped recording + sorting.

    Function-scoped — each test gets its own NWBFile so writes don't leak
    across tests, while the underlying recording/sorting/analyzer are shared.
    """
    from datetime import datetime, timezone

    from neuroconv.tools.spikeinterface import add_recording_to_nwbfile, add_sorting_to_nwbfile
    from pynwb import NWBFile

    _, recording, sorting = sorting_analyzer_with_extensions

    nwbfile = NWBFile(
        session_description="Test",
        identifier=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        session_start_time=datetime.now(timezone.utc),
    )
    add_recording_to_nwbfile(recording, nwbfile=nwbfile, write_as="raw", iterator_type=None)
    add_sorting_to_nwbfile(sorting, nwbfile=nwbfile, write_as="units")
    return nwbfile

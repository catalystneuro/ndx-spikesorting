"""Class-level tests for canonical typed VectorData column types.

One ``Test<ClassName>`` per canonical type. Tests are independent of the
SortingAnalyzer pipeline.
"""

import numpy as np
import pytest

from pynwb import NWBHDF5IO, read_nwb
from pynwb.testing.mock.ecephys import mock_Units
from pynwb.testing.mock.file import mock_NWBFile

from ndx_spikesorting import FiringRate, NumSpikes


@pytest.fixture
def nwbfile_with_units():
    """NWBFile with a populated Units table from pynwb's mocks."""
    nwbfile = mock_NWBFile()
    mock_Units(num_units=3, nwbfile=nwbfile)
    return nwbfile


class TestFiringRate:
    def test_construct(self):
        fr = FiringRate(name="firing_rate", description="rates", data=[1.0, 2.0, 3.0])
        assert fr.name == "firing_rate"
        np.testing.assert_array_equal(fr.data, [1.0, 2.0, 3.0])

    def test_unit_attribute(self):
        fr = FiringRate(name="firing_rate", description="rates", data=[1.0])
        assert fr.unit == "hertz"

    def test_added_to_units_table(self, nwbfile_with_units):
        nwbfile_with_units.units.add_column(
            name="firing_rate",
            description="rates",
            data=[10.0, 20.0, 30.0],
            col_cls=FiringRate,
        )
        assert isinstance(nwbfile_with_units.units["firing_rate"], FiringRate)

    def test_round_trip_through_nwb_io(self, nwbfile_with_units, tmp_path):
        nwbfile_with_units.units.add_column(
            name="firing_rate",
            description="rates",
            data=[10.0, 20.0, 30.0],
            col_cls=FiringRate,
        )

        path = tmp_path / "firing_rate.nwb"
        with NWBHDF5IO(path, mode="w") as io:
            io.write(nwbfile_with_units)

        loaded = read_nwb(path=path)
        assert isinstance(loaded.units["firing_rate"], FiringRate)
        np.testing.assert_array_equal(loaded.units["firing_rate"].data[:], [10.0, 20.0, 30.0])
        assert loaded.units["firing_rate"].unit == "hertz"


class TestNumSpikes:
    def test_construct(self):
        ns = NumSpikes(
            name="num_spikes",
            description="counts",
            data=np.array([10, 20, 30], dtype=np.int64),
        )
        assert ns.name == "num_spikes"
        np.testing.assert_array_equal(ns.data, [10, 20, 30])

    def test_added_to_units_table(self, nwbfile_with_units):
        nwbfile_with_units.units.add_column(
            name="num_spikes",
            description="counts",
            data=np.array([10, 20, 30], dtype=np.int64),
            col_cls=NumSpikes,
        )
        assert isinstance(nwbfile_with_units.units["num_spikes"], NumSpikes)

    def test_round_trip_through_nwb_io(self, nwbfile_with_units, tmp_path):
        nwbfile_with_units.units.add_column(
            name="num_spikes",
            description="counts",
            data=np.array([10, 20, 30], dtype=np.int64),
            col_cls=NumSpikes,
        )

        path = tmp_path / "num_spikes.nwb"
        with NWBHDF5IO(path, mode="w") as io:
            io.write(nwbfile_with_units)

        loaded = read_nwb(path=path)
        assert isinstance(loaded.units["num_spikes"], NumSpikes)
        np.testing.assert_array_equal(loaded.units["num_spikes"].data[:], [10, 20, 30])

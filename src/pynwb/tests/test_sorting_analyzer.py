"""Unit and integration tests for the ndx-spikesorting extension types."""

import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.testing.mock.device import mock_Device
from pynwb.testing.mock.ecephys import mock_ElectrodeGroup, mock_ElectrodeTable
from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing import TestCase, remove_test_file, NWBH5IOFlexMixin
from hdmf.common import VectorData, VectorIndex, DynamicTableRegion

from ndx_spikesorting import (
    RandomSpikesData,
    TemplatesData,
    SortingAnalyzerExtensions,
    SortingAnalyzerContainer,
)


def set_up_nwbfile(nwbfile: NWBFile = None, n_electrodes: int = 10, n_units: int = 3):
    """Create an NWBFile with a Device, ElectrodeGroup, electrodes, and units."""
    nwbfile = nwbfile or mock_NWBFile()
    device = mock_Device(nwbfile=nwbfile)
    electrode_group = mock_ElectrodeGroup(device=device, nwbfile=nwbfile)
    _ = mock_ElectrodeTable(n_rows=n_electrodes, group=electrode_group, nwbfile=nwbfile)

    # Add units with mock spike times
    for i in range(n_units):
        spike_times = np.sort(np.random.uniform(0, 10, size=50))  # Random spike times
        nwbfile.add_unit(spike_times=spike_times)

    return nwbfile


def create_units_region(nwbfile: NWBFile, n_units: int = 3):
    """Create a DynamicTableRegion for the units table."""
    return DynamicTableRegion(
        name="units_region",
        data=list(range(n_units)),
        description="All units from sorting analysis",
        table=nwbfile.units,
    )


def create_mock_random_spikes_data(num_units: int = 3, spikes_per_unit: list = None):
    """Create mock RandomSpikesData with ragged spike indices."""
    if spikes_per_unit is None:
        spikes_per_unit = [10, 15, 8]  # Different number of spikes per unit

    # Create concatenated indices
    all_indices = []
    cumulative_index = []
    for n_spikes in spikes_per_unit:
        unit_indices = np.sort(np.random.choice(1000, size=n_spikes, replace=False))
        all_indices.append(unit_indices)
        cumulative_index.append(len(np.concatenate(all_indices)))

    random_spikes_indices_data = np.concatenate(all_indices).astype(np.int64)
    index_data = np.array(cumulative_index, dtype=np.int64)

    # Create VectorData for the indices (this is the target for VectorIndex)
    random_spikes_indices = VectorData(
        name="random_spikes_indices",
        data=random_spikes_indices_data,
        description="Random spike indices for all units",
    )

    # Create VectorIndex pointing to the VectorData
    random_spikes_indices_index = VectorIndex(
        name="random_spikes_indices_index",
        data=index_data,
        target=random_spikes_indices,
    )

    random_spikes = RandomSpikesData(
        name="random_spikes",
        method="uniform",
        max_spikes_per_unit=500,
        seed=42,
        random_spikes_indices=random_spikes_indices,
        random_spikes_indices_index=random_spikes_indices_index,
    )
    return random_spikes


def create_mock_templates_data(num_units: int = 3, num_samples: int = 60, channels_per_unit: list = None):
    """Create mock TemplatesData with sparse templates."""
    if channels_per_unit is None:
        # Simulate sparse channels: each unit has different active channels
        channels_per_unit = [[0, 1, 2], [1, 2, 3, 4], [2, 3]]  # Variable channels per unit

    # Build sparse template data
    all_data = []
    all_channel_ids = []
    cumulative_index = []

    for unit_channels in channels_per_unit:
        n_channels = len(unit_channels)
        # Random template waveforms for this unit's active channels
        unit_templates = np.random.randn(n_channels, num_samples).astype(np.float32)
        all_data.append(unit_templates)
        all_channel_ids.extend(unit_channels)
        cumulative_index.append(len(np.vstack(all_data)))

    data_array = np.vstack(all_data).astype(np.float32)
    channel_ids = np.array(all_channel_ids, dtype=np.int32)
    index_data = np.array(cumulative_index, dtype=np.int64)

    # Create VectorData for the template data (this is the target for VectorIndex)
    data = VectorData(
        name="data",
        data=data_array,
        description="Sparse template waveforms",
    )

    # Create VectorIndex pointing to the VectorData
    data_index = VectorIndex(
        name="data_index",
        data=index_data,
        target=data,
    )

    templates = TemplatesData(
        name="templates",
        peak_sample_index=20,
        data=data,
        data_index=data_index,
        channel_ids=channel_ids,
    )
    return templates


class TestRandomSpikesDataConstructor(TestCase):
    """Unit tests for RandomSpikesData constructor."""

    def test_constructor(self):
        """Test that RandomSpikesData constructor sets values correctly."""
        random_spikes = create_mock_random_spikes_data()

        self.assertEqual(random_spikes.name, "random_spikes")
        self.assertEqual(random_spikes.method, "uniform")
        self.assertEqual(random_spikes.max_spikes_per_unit, 500)
        self.assertEqual(random_spikes.seed, 42)
        self.assertEqual(len(random_spikes.random_spikes_indices.data), 33)  # 10 + 15 + 8

    def test_constructor_all_method(self):
        """Test RandomSpikesData with 'all' method."""
        indices_data = np.arange(100, dtype=np.int64)
        index_data = np.array([100], dtype=np.int64)

        random_spikes_indices = VectorData(
            name="random_spikes_indices",
            data=indices_data,
            description="All spike indices",
        )

        random_spikes = RandomSpikesData(
            name="random_spikes",
            method="all",
            random_spikes_indices=random_spikes_indices,
            random_spikes_indices_index=VectorIndex(
                name="random_spikes_indices_index",
                data=index_data,
                target=random_spikes_indices,
            ),
        )

        self.assertEqual(random_spikes.method, "all")


class TestTemplatesDataConstructor(TestCase):
    """Unit tests for TemplatesData constructor."""

    def test_constructor(self):
        """Test that TemplatesData constructor sets values correctly."""
        templates = create_mock_templates_data()

        self.assertEqual(templates.name, "templates")
        self.assertEqual(templates.peak_sample_index, 20)
        self.assertEqual(templates.data.data.shape[1], 60)  # num_samples
        # Total channels: 3 + 4 + 2 = 9
        self.assertEqual(templates.data.data.shape[0], 9)
        self.assertEqual(len(templates.channel_ids), 9)


class TestSortingAnalyzerContainerConstructor(TestCase):
    """Unit tests for SortingAnalyzerContainer constructor."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()

    def test_constructor_minimal(self):
        """Test SortingAnalyzerContainer with minimal required fields."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        container = SortingAnalyzerContainer(
            name="sorting_analyzer",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
        )

        self.assertEqual(container.name, "sorting_analyzer")
        self.assertEqual(container.sampling_frequency, 30000.0)

    def test_constructor_with_sparsity(self):
        """Test SortingAnalyzerContainer with sparsity mask."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        num_units = 3  # Match the units added in set_up_nwbfile
        num_channels = 10
        sparsity_mask = np.random.rand(num_units, num_channels) > 0.5

        container = SortingAnalyzerContainer(
            name="sorting_analyzer",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
            sparsity_mask=sparsity_mask,
        )

        np.testing.assert_array_equal(container.sparsity_mask, sparsity_mask)


class TestRandomSpikesDataRoundtrip(TestCase):
    """Roundtrip test for RandomSpikesData."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_random_spikes.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading RandomSpikesData."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        random_spikes = create_mock_random_spikes_data()

        # Create extensions container
        extensions = SortingAnalyzerExtensions(name="extensions")
        extensions.random_spikes_data = random_spikes

        # Create main container
        container = SortingAnalyzerContainer(
            name="sorting_analyzer",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
        )
        container.sorting_analyzer_extensions = extensions

        # Add to processing module
        ecephys_module = self.nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing",
        )
        ecephys_module.add(container)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_container = read_nwbfile.processing["ecephys"]["sorting_analyzer"]
            read_extensions = read_container.sorting_analyzer_extensions
            read_random_spikes = read_extensions.random_spikes_data

            self.assertEqual(read_random_spikes.method, "uniform")
            self.assertEqual(read_random_spikes.max_spikes_per_unit, 500)
            self.assertEqual(read_random_spikes.seed, 42)
            np.testing.assert_array_equal(
                read_random_spikes.random_spikes_indices.data[:],
                random_spikes.random_spikes_indices.data[:],
            )


class TestTemplatesDataRoundtrip(TestCase):
    """Roundtrip test for TemplatesData."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_templates.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading TemplatesData."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        templates = create_mock_templates_data()

        # Create extensions container
        extensions = SortingAnalyzerExtensions(name="extensions")
        extensions.templates_data = templates

        # Create main container
        container = SortingAnalyzerContainer(
            name="sorting_analyzer",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
        )
        container.sorting_analyzer_extensions = extensions

        ecephys_module = self.nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing",
        )
        ecephys_module.add(container)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_container = read_nwbfile.processing["ecephys"]["sorting_analyzer"]
            read_extensions = read_container.sorting_analyzer_extensions
            read_templates = read_extensions.templates_data

            self.assertEqual(read_templates.peak_sample_index, 20)
            np.testing.assert_array_almost_equal(
                read_templates.data.data[:],
                templates.data.data[:],
            )
            np.testing.assert_array_equal(
                read_templates.channel_ids[:],
                templates.channel_ids,
            )


class TestSortingAnalyzerContainerRoundtrip(TestCase):
    """Full roundtrip test for SortingAnalyzerContainer with all components."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_sorting_analyzer.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip_full(self):
        """Test writing and reading SortingAnalyzerContainer with all extensions."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        random_spikes = create_mock_random_spikes_data()
        templates = create_mock_templates_data()

        num_units = 3
        num_channels = 10
        sparsity_mask = np.zeros((num_units, num_channels), dtype=bool)
        # Set sparsity based on channels_per_unit used in templates
        sparsity_mask[0, [0, 1, 2]] = True
        sparsity_mask[1, [1, 2, 3, 4]] = True
        sparsity_mask[2, [2, 3]] = True

        # Create extensions container with both extensions
        extensions = SortingAnalyzerExtensions(name="extensions")
        extensions.random_spikes_data = random_spikes
        extensions.templates_data = templates

        # Create main container with sparsity and extensions
        container = SortingAnalyzerContainer(
            name="sorting_analyzer",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
            sparsity_mask=sparsity_mask,
        )
        container.sorting_analyzer_extensions = extensions

        ecephys_module = self.nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing",
        )
        ecephys_module.add(container)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_container = read_nwbfile.processing["ecephys"]["sorting_analyzer"]

            # Verify main container
            self.assertEqual(read_container.sampling_frequency, 30000.0)
            np.testing.assert_array_equal(read_container.sparsity_mask[:], sparsity_mask)

            # Verify units_region
            self.assertEqual(len(read_container.units_region.data[:]), 3)

            # Verify extensions container
            read_extensions = read_container.sorting_analyzer_extensions

            # Verify random spikes
            read_random_spikes = read_extensions.random_spikes_data
            self.assertEqual(read_random_spikes.method, "uniform")
            self.assertEqual(read_random_spikes.max_spikes_per_unit, 500)

            # Verify templates
            read_templates = read_extensions.templates_data
            self.assertEqual(read_templates.peak_sample_index, 20)


class TestSortingAnalyzerContainerRoundtripPyNWB(NWBH5IOFlexMixin, TestCase):
    """Complex roundtrip test using pynwb.testing infrastructure."""

    def getContainerType(self):
        return "SortingAnalyzerContainer"

    def addContainer(self):
        set_up_nwbfile(self.nwbfile)

        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        container = SortingAnalyzerContainer(
            name="sorting_analyzer",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
        )

        ecephys_module = self.nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing",
        )
        ecephys_module.add(container)

    def getContainer(self, nwbfile: NWBFile):
        return nwbfile.processing["ecephys"]["sorting_analyzer"]

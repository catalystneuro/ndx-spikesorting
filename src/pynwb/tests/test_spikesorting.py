"""Unit and integration tests for the ndx-spikesorting extension types."""

import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.testing.mock.device import mock_Device
from pynwb.testing.mock.ecephys import mock_ElectrodeGroup, mock_ElectrodeTable
from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing import TestCase, remove_test_file, NWBH5IOFlexMixin
from hdmf.common import VectorData, VectorIndex, DynamicTableRegion

from ndx_spikesorting import (
    RandomSpikes,
    Templates,
    NoiseLevels,
    SpikeSortingExtensions,
    SpikeSortingContainer,
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


def create_mock_random_spikes(num_units: int = 3, spikes_per_unit: list = None):
    """Create mock RandomSpikes with ragged spike indices."""
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

    random_spikes_indices = VectorData(
        name="random_spikes_indices",
        data=random_spikes_indices_data,
        description="Random spike indices for all units",
    )

    random_spikes_indices_index = VectorIndex(
        name="random_spikes_indices_index",
        data=index_data,
        target=random_spikes_indices,
    )

    random_spikes = RandomSpikes(
        name="random_spikes",
        random_spikes_indices=random_spikes_indices,
        random_spikes_indices_index=random_spikes_indices_index,
    )
    return random_spikes

def create_mock_noise_levels(num_channels: int = 120):
    """Create mock NoiseLevels with random noise values."""
    noise_levels_data = VectorData(
        name="data",
        data=np.random.rand(num_channels).astype(np.float32),
        description="Random noise levels for all channels",
    )

    noise_levels = NoiseLevels(
        name="noise_levels",
        data=noise_levels_data,
    )
    return noise_levels

def create_mock_templates(nwbfile: NWBFile, num_units: int = 3, num_samples: int = 60, channels_per_unit: list = None):
    """Create mock Templates with sparse templates and electrode references."""
    if channels_per_unit is None:
        channels_per_unit = [[0, 1, 2], [1, 2, 3, 4], [2, 3]]

    all_data = []
    all_electrode_indices = []
    cumulative_index = []

    for unit_channels in channels_per_unit:
        n_channels = len(unit_channels)
        unit_templates = np.random.randn(n_channels, num_samples).astype(np.float32)
        all_data.append(unit_templates)
        all_electrode_indices.extend(unit_channels)
        cumulative_index.append(len(np.vstack(all_data)))

    data_array = np.vstack(all_data).astype(np.float32)
    index_data = np.array(cumulative_index, dtype=np.int64)

    data = VectorData(
        name="data",
        data=data_array,
        description="Template waveforms",
    )

    data_index = VectorIndex(
        name="data_index",
        data=index_data,
        target=data,
    )

    electrodes = DynamicTableRegion(
        name="electrodes",
        data=list(int(i) for i in all_electrode_indices),
        description="Electrode for each waveform row.",
        table=nwbfile.electrodes,
    )

    templates = Templates(
        name="templates",
        peak_sample_index=20,
        data=data,
        data_index=data_index,
        electrodes=electrodes,
    )
    return templates


class TestRandomSpikesConstructor(TestCase):
    """Unit tests for RandomSpikes constructor."""

    def test_constructor(self):
        """Test that RandomSpikes constructor sets values correctly."""
        random_spikes = create_mock_random_spikes()

        self.assertEqual(random_spikes.name, "random_spikes")
        self.assertEqual(len(random_spikes.random_spikes_indices.data), 33)  # 10 + 15 + 8

class TestNoiseLevelsConstructor(TestCase):
    """Unit tests for NoiseLevels constructor."""

    def test_constructor(self):
        """Test that NoiseLevels constructor sets values correctly."""
        noise_levels = create_mock_noise_levels()

        self.assertEqual(noise_levels.name, "noise_levels")
        self.assertEqual(noise_levels.data.shape[0], 120)

class TestTemplatesConstructor(TestCase):
    """Unit tests for Templates constructor."""

    def test_constructor(self):
        """Test that Templates constructor sets values correctly."""
        nwbfile = set_up_nwbfile()
        templates = create_mock_templates(nwbfile)

        self.assertEqual(templates.name, "templates")
        self.assertEqual(templates.peak_sample_index, 20)
        self.assertEqual(templates.data.data.shape[1], 60)  # num_samples
        # Total channels: 3 + 4 + 2 = 9
        self.assertEqual(templates.data.data.shape[0], 9)
        self.assertEqual(len(templates.electrodes.data), 9)


class TestSpikeSortingContainerConstructor(TestCase):
    """Unit tests for SpikeSortingContainer constructor."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()

    def test_constructor_minimal(self):
        """Test SpikeSortingContainer with minimal required fields."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        container = SpikeSortingContainer(
            name="spike_sorting",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
        )

        self.assertEqual(container.name, "spike_sorting")
        self.assertEqual(container.sampling_frequency, 30000.0)

    def test_constructor_with_sparsity(self):
        """Test SpikeSortingContainer with sparsity mask."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        num_units = 3  # Match the units added in set_up_nwbfile
        num_channels = 10
        sparsity_mask = np.random.rand(num_units, num_channels) > 0.5

        container = SpikeSortingContainer(
            name="spike_sorting",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
            sparsity_mask=sparsity_mask,
        )

        np.testing.assert_array_equal(container.sparsity_mask, sparsity_mask)


class TestRandomSpikesRoundtrip(TestCase):
    """Roundtrip test for RandomSpikes."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_random_spikes.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading RandomSpikes."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        random_spikes = create_mock_random_spikes()

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.random_spikes = random_spikes

        container = SpikeSortingContainer(
            name="spike_sorting",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
        )
        container.spike_sorting_extensions = extensions

        ecephys_module = self.nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing",
        )
        ecephys_module.add(container)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_container = read_nwbfile.processing["ecephys"]["spike_sorting"]
            read_extensions = read_container.spike_sorting_extensions
            read_random_spikes = read_extensions.random_spikes

            np.testing.assert_array_equal(
                read_random_spikes.random_spikes_indices.data[:],
                random_spikes.random_spikes_indices.data[:],
            )

class TestNoiseLevelsRoundtrip(TestCase):
    """Roundtrip test for NoiseLevels."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_noise_levels.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading NoiseLevels."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        noise_levels = create_mock_noise_levels()

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.noise_levels = noise_levels

        container = SpikeSortingContainer(
            name="spike_sorting",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
        )
        container.spike_sorting_extensions = extensions

        ecephys_module = self.nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing",
        )
        ecephys_module.add(container)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_container = read_nwbfile.processing["ecephys"]["spike_sorting"]
            read_extensions = read_container.spike_sorting_extensions
            read_noise_levels = read_extensions.noise_levels

            np.testing.assert_array_equal(
                read_noise_levels.data[:],
                noise_levels.data[:],
            )

class TestTemplatesRoundtrip(TestCase):
    """Roundtrip test for Templates."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_templates.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading Templates."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        templates = create_mock_templates(self.nwbfile)

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.templates = templates

        container = SpikeSortingContainer(
            name="spike_sorting",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
        )
        container.spike_sorting_extensions = extensions

        ecephys_module = self.nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing",
        )
        ecephys_module.add(container)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_container = read_nwbfile.processing["ecephys"]["spike_sorting"]
            read_extensions = read_container.spike_sorting_extensions
            read_templates = read_extensions.templates

            self.assertEqual(read_templates.peak_sample_index, 20)
            np.testing.assert_array_almost_equal(
                read_templates.data.data[:],
                templates.data.data[:],
            )
            np.testing.assert_array_equal(
                read_templates.electrodes.data[:],
                templates.electrodes.data[:],
            )


class TestSpikeSortingContainerRoundtrip(TestCase):
    """Full roundtrip test for SpikeSortingContainer with all components."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_spike_sorting.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip_full(self):
        """Test writing and reading SpikeSortingContainer with all extensions."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        random_spikes = create_mock_random_spikes()
        noise_levels = create_mock_noise_levels()
        templates = create_mock_templates(self.nwbfile)

        num_units = 3
        num_channels = 10
        sparsity_mask = np.zeros((num_units, num_channels), dtype=bool)
        sparsity_mask[0, [0, 1, 2]] = True
        sparsity_mask[1, [1, 2, 3, 4]] = True
        sparsity_mask[2, [2, 3]] = True

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.random_spikes = random_spikes
        extensions.templates = templates
        extensions.noise_levels = noise_levels
        container = SpikeSortingContainer(
            name="spike_sorting",
            sampling_frequency=30000.0,
            electrodes=electrodes_region,
            units_region=units_region,
            sparsity_mask=sparsity_mask,
        )
        container.spike_sorting_extensions = extensions

        ecephys_module = self.nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing",
        )
        ecephys_module.add(container)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwbfile = io.read()
            read_container = read_nwbfile.processing["ecephys"]["spike_sorting"]

            self.assertEqual(read_container.sampling_frequency, 30000.0)
            np.testing.assert_array_equal(read_container.sparsity_mask[:], sparsity_mask)

            self.assertEqual(len(read_container.units_region.data[:]), 3)

            read_extensions = read_container.spike_sorting_extensions

            read_random_spikes = read_extensions.random_spikes
            self.assertIsNotNone(read_random_spikes)

            read_templates = read_extensions.templates
            self.assertEqual(read_templates.peak_sample_index, 20)

            read_noise_levels = read_extensions.noise_levels
            self.assertIsNotNone(read_noise_levels)


class TestSpikeSortingContainerRoundtripPyNWB(NWBH5IOFlexMixin, TestCase):
    """Complex roundtrip test using pynwb.testing infrastructure."""

    def getContainerType(self):
        return "SpikeSortingContainer"

    def addContainer(self):
        set_up_nwbfile(self.nwbfile)

        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        container = SpikeSortingContainer(
            name="spike_sorting",
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
        return nwbfile.processing["ecephys"]["spike_sorting"]

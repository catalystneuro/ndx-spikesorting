"""Unit and integration tests for the ndx-spikesorting extension types."""

import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.testing.mock.device import mock_Device
from pynwb.testing.mock.ecephys import mock_ElectrodeGroup, mock_ElectrodesTable
from pynwb.testing.mock.file import mock_NWBFile
from pynwb.testing import TestCase, remove_test_file, NWBH5IOFlexMixin
from hdmf.common import VectorData, VectorIndex, DynamicTableRegion

from ndx_spikesorting import (
    RandomSpikes,
    Templates,
    NoiseLevels,
    UnitLocations,
    Correlograms,
    ISIHistograms,
    TemplateSimilarity,
    SpikeAmplitudes,
    AmplitudeScalings,
    SpikeLocations,
    SpikeSortingExtensions,
    SpikeSortingContainer,
)


def set_up_nwbfile(nwbfile: NWBFile = None, n_electrodes: int = 10, n_units: int = 3):
    """Create an NWBFile with a Device, ElectrodeGroup, electrodes, and units."""
    nwbfile = nwbfile or mock_NWBFile()
    device = mock_Device(nwbfile=nwbfile)
    electrode_group = mock_ElectrodeGroup(device=device, nwbfile=nwbfile)
    _ = mock_ElectrodesTable(n_rows=n_electrodes, group=electrode_group, nwbfile=nwbfile)

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

    noise_levels = NoiseLevels(
        name="noise_levels",
        data=np.random.rand(num_channels),
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

def create_mock_unit_locations_2d(nwbfile: NWBFile, num_units: int = 3):
    """Create mock UnitLocations with random 2D coordinates for each unit."""
    locations_data = np.random.rand(num_units, 2).astype(np.float32)  # Random (x, y) coordinates

    unit_locations = UnitLocations(
        name="unit_locations",
        data=locations_data,
    )
    return unit_locations

def create_mock_unit_locations_3d(nwbfile: NWBFile, num_units: int = 3):
    """Create mock UnitLocations with random 3D coordinates for each unit."""
    locations_data = np.random.rand(num_units, 3).astype(np.float32)  # Random (x, y, z) coordinates

    unit_locations = UnitLocations(
        name="unit_locations",
        data=locations_data,
    )
    return unit_locations


def create_mock_correlograms(num_units: int = 3, num_bins: int = 50):
    """Create mock Correlograms with random data."""
    data = np.random.randint(0, 100, size=(num_units, num_units, num_bins))
    bin_edges = np.linspace(-50, 50, num_bins).astype(np.float64)

    correlograms = Correlograms(
        name="correlograms",
        data=data,
        bin_edges=bin_edges,
    )
    return correlograms


def create_mock_isi_histograms(num_units: int = 3, num_bins: int = 100):
    """Create mock ISIHistograms with random data."""
    data = np.random.randint(0, 100, size=(num_units, num_bins))
    bin_edges = np.linspace(0, 100, num_bins).astype(np.float64)

    isi_histograms = ISIHistograms(
        name="isi_histograms",
        data=data,
        bin_edges=bin_edges,
    )
    return isi_histograms


def create_mock_template_similarity(num_units: int = 3):
    """Create mock TemplateSimilarity with random similarity matrix."""
    data = np.random.rand(num_units, num_units).astype(np.float64)
    # Make symmetric
    data = (data + data.T) / 2.0
    np.fill_diagonal(data, 1.0)

    template_similarity = TemplateSimilarity(
        name="template_similarity",
        data=data,
    )
    return template_similarity


def create_mock_spike_amplitudes(num_units: int = 3, spikes_per_unit: list = None):
    """Create mock SpikeAmplitudes with ragged amplitude data."""
    if spikes_per_unit is None:
        spikes_per_unit = [50, 40, 60]

    all_amplitudes = []
    cumulative_index = []
    for n_spikes in spikes_per_unit:
        unit_amplitudes = np.random.randn(n_spikes).astype(np.float64) * 100
        all_amplitudes.append(unit_amplitudes)
        cumulative_index.append(sum(len(a) for a in all_amplitudes))

    data = VectorData(
        name="data",
        data=np.concatenate(all_amplitudes),
        description="Spike amplitudes for all units",
    )
    data_index = VectorIndex(
        name="data_index",
        data=np.array(cumulative_index, dtype=np.int64),
        target=data,
    )

    spike_amplitudes = SpikeAmplitudes(
        name="spike_amplitudes",
        data=data,
        data_index=data_index,
    )
    return spike_amplitudes


def create_mock_spike_locations(num_units: int = 3, spikes_per_unit: list = None, ndim: int = 2):
    """Create mock SpikeLocations with ragged location data."""
    if spikes_per_unit is None:
        spikes_per_unit = [50, 40, 60]

    all_locations = []
    cumulative_index = []
    for n_spikes in spikes_per_unit:
        unit_locations = np.random.rand(n_spikes, ndim).astype(np.float64)
        all_locations.append(unit_locations)
        cumulative_index.append(sum(len(a) for a in all_locations))

    data = VectorData(
        name="data",
        data=np.vstack(all_locations),
        description="Spike locations for all units",
    )
    data_index = VectorIndex(
        name="data_index",
        data=np.array(cumulative_index, dtype=np.int64),
        target=data,
    )

    spike_locations = SpikeLocations(
        name="spike_locations",
        data=data,
        data_index=data_index,
    )
    return spike_locations


def create_mock_amplitude_scalings(num_units: int = 3, spikes_per_unit: list = None):
    """Create mock AmplitudeScalings with ragged scaling data."""
    if spikes_per_unit is None:
        spikes_per_unit = [50, 40, 60]

    all_scalings = []
    cumulative_index = []
    for n_spikes in spikes_per_unit:
        unit_scalings = np.random.rand(n_spikes).astype(np.float32) + 0.5
        all_scalings.append(unit_scalings)
        cumulative_index.append(sum(len(a) for a in all_scalings))

    data = VectorData(
        name="data",
        data=np.concatenate(all_scalings),
        description="Amplitude scalings for all units",
    )
    data_index = VectorIndex(
        name="data_index",
        data=np.array(cumulative_index, dtype=np.int64),
        target=data,
    )

    amplitude_scalings = AmplitudeScalings(
        name="amplitude_scalings",
        data=data,
        data_index=data_index,
    )
    return amplitude_scalings


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


class TestUnitLocationsConstructor(TestCase):
    """Unit tests for UnitLocations constructor."""

    def test_constructor_2d(self):
        """Test that UnitLocations constructor sets 2D locations correctly."""
        nwbfile = set_up_nwbfile()
        unit_locations = create_mock_unit_locations_2d(nwbfile)

        self.assertEqual(unit_locations.name, "unit_locations")
        self.assertEqual(unit_locations.data.shape, (3, 2))  # 3 units, 2D coordinates

    def test_constructor_3d(self):
        """Test that UnitLocations constructor sets 3D locations correctly."""
        nwbfile = set_up_nwbfile()
        unit_locations = create_mock_unit_locations_3d(nwbfile)

        self.assertEqual(unit_locations.name, "unit_locations")
        self.assertEqual(unit_locations.data.shape, (3, 3))  # 3 units, 3D coordinates

class TestCorrelogramsConstructor(TestCase):
    """Unit tests for Correlograms constructor."""

    def test_constructor(self):
        """Test that Correlograms constructor sets values correctly."""
        correlograms = create_mock_correlograms(num_units=3, num_bins=50)

        self.assertEqual(correlograms.name, "correlograms")
        self.assertEqual(correlograms.data.shape, (3, 3, 50))
        self.assertEqual(correlograms.bin_edges.shape, (50,))


class TestISIHistogramsConstructor(TestCase):
    """Unit tests for ISIHistograms constructor."""

    def test_constructor(self):
        """Test that ISIHistograms constructor sets values correctly."""
        isi_histograms = create_mock_isi_histograms(num_units=3, num_bins=100)

        self.assertEqual(isi_histograms.name, "isi_histograms")
        self.assertEqual(isi_histograms.data.shape, (3, 100))
        self.assertEqual(isi_histograms.bin_edges.shape, (100,))


class TestTemplateSimilarityConstructor(TestCase):
    """Unit tests for TemplateSimilarity constructor."""

    def test_constructor(self):
        """Test that TemplateSimilarity constructor sets values correctly."""
        template_similarity = create_mock_template_similarity(num_units=3)

        self.assertEqual(template_similarity.name, "template_similarity")
        self.assertEqual(template_similarity.data.shape, (3, 3))


class TestSpikeAmplitudesConstructor(TestCase):
    """Unit tests for SpikeAmplitudes constructor."""

    def test_constructor(self):
        """Test that SpikeAmplitudes constructor sets values correctly."""
        spike_amplitudes = create_mock_spike_amplitudes()

        self.assertEqual(spike_amplitudes.name, "spike_amplitudes")
        self.assertEqual(len(spike_amplitudes.data.data), 150)  # 50 + 40 + 60


class TestSpikeLocationsConstructor(TestCase):
    """Unit tests for SpikeLocations constructor."""

    def test_constructor_2d(self):
        """Test that SpikeLocations constructor sets 2D locations correctly."""
        spike_locations = create_mock_spike_locations(ndim=2)

        self.assertEqual(spike_locations.name, "spike_locations")
        self.assertEqual(spike_locations.data.data.shape, (150, 2))

    def test_constructor_3d(self):
        """Test that SpikeLocations constructor sets 3D locations correctly."""
        spike_locations = create_mock_spike_locations(ndim=3)

        self.assertEqual(spike_locations.name, "spike_locations")
        self.assertEqual(spike_locations.data.data.shape, (150, 3))


class TestAmplitudeScalingsConstructor(TestCase):
    """Unit tests for AmplitudeScalings constructor."""

    def test_constructor(self):
        """Test that AmplitudeScalings constructor sets values correctly."""
        amplitude_scalings = create_mock_amplitude_scalings()

        self.assertEqual(amplitude_scalings.name, "amplitude_scalings")
        self.assertEqual(len(amplitude_scalings.data.data), 150)  # 50 + 40 + 60


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


class TestUnitLocationsRoundtrip(TestCase):
    """Roundtrip test for UnitLocations."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_unit_locations.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading UnitLocations."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        unit_locations = create_mock_unit_locations_3d(self.nwbfile)

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.unit_locations = unit_locations

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
            read_unit_locations = read_extensions.unit_locations

            np.testing.assert_array_almost_equal(
                read_unit_locations.data[:],
                unit_locations.data[:],
            )


class TestCorrelogramsRoundtrip(TestCase):
    """Roundtrip test for Correlograms."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_correlograms.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading Correlograms."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        correlograms = create_mock_correlograms()

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.correlograms = correlograms

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
            read_correlograms = read_extensions.correlograms

            np.testing.assert_array_almost_equal(
                read_correlograms.data[:],
                correlograms.data[:],
            )
            np.testing.assert_array_almost_equal(
                read_correlograms.bin_edges[:],
                correlograms.bin_edges[:],
            )


class TestISIHistogramsRoundtrip(TestCase):
    """Roundtrip test for ISIHistograms."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_isi_histograms.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading ISIHistograms."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        isi_histograms = create_mock_isi_histograms()

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.isi_histograms = isi_histograms

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
            read_isi_histograms = read_extensions.isi_histograms

            np.testing.assert_array_almost_equal(
                read_isi_histograms.data[:],
                isi_histograms.data[:],
            )
            np.testing.assert_array_almost_equal(
                read_isi_histograms.bin_edges[:],
                isi_histograms.bin_edges[:],
            )


class TestTemplateSimilarityRoundtrip(TestCase):
    """Roundtrip test for TemplateSimilarity."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_template_similarity.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading TemplateSimilarity."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        template_similarity = create_mock_template_similarity()

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.template_similarity = template_similarity

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
            read_template_similarity = read_extensions.template_similarity

            np.testing.assert_array_almost_equal(
                read_template_similarity.data[:],
                template_similarity.data[:],
            )


class TestSpikeAmplitudesRoundtrip(TestCase):
    """Roundtrip test for SpikeAmplitudes."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_spike_amplitudes.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading SpikeAmplitudes."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        spike_amplitudes = create_mock_spike_amplitudes()

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.spike_amplitudes = spike_amplitudes

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
            read_spike_amplitudes = read_extensions.spike_amplitudes

            np.testing.assert_array_almost_equal(
                read_spike_amplitudes.data.data[:],
                spike_amplitudes.data.data[:],
            )
            np.testing.assert_array_equal(
                read_spike_amplitudes.data_index.data[:],
                spike_amplitudes.data_index.data[:],
            )


class TestSpikeLocationsRoundtrip(TestCase):
    """Roundtrip test for SpikeLocations."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_spike_locations.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip_2d(self):
        """Test writing and reading SpikeLocations with 2D data."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        spike_locations = create_mock_spike_locations(ndim=2)

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.spike_locations = spike_locations

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
            read_spike_locations = read_extensions.spike_locations

            np.testing.assert_array_almost_equal(
                read_spike_locations.data.data[:],
                spike_locations.data.data[:],
            )
            np.testing.assert_array_equal(
                read_spike_locations.data_index.data[:],
                spike_locations.data_index.data[:],
            )

    def test_roundtrip_3d(self):
        """Test writing and reading SpikeLocations with 3D data."""
        self.nwbfile = set_up_nwbfile()

        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        spike_locations = create_mock_spike_locations(ndim=3)

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.spike_locations = spike_locations

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
            read_spike_locations = read_extensions.spike_locations

            np.testing.assert_array_almost_equal(
                read_spike_locations.data.data[:],
                spike_locations.data.data[:],
            )


class TestAmplitudeScalingsRoundtrip(TestCase):
    """Roundtrip test for AmplitudeScalings."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_amplitude_scalings.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading AmplitudeScalings."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        amplitude_scalings = create_mock_amplitude_scalings()

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.amplitude_scalings = amplitude_scalings

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
            read_amplitude_scalings = read_extensions.amplitude_scalings

            np.testing.assert_array_almost_equal(
                read_amplitude_scalings.data.data[:],
                amplitude_scalings.data.data[:],
            )
            np.testing.assert_array_equal(
                read_amplitude_scalings.data_index.data[:],
                amplitude_scalings.data_index.data[:],
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

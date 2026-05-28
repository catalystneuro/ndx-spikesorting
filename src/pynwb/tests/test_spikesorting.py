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
    Waveforms,
    Templates,
    NoiseLevels,
    UnitLocations,
    Correlograms,
    ISIHistograms,
    TemplateSimilarity,
    SpikeAmplitudes,
    AmplitudeScalings,
    SpikeLocations,
    PCAProjectionsByChannel,
    PCAProjectionsConcatenated,
    UnitMetrics,
    ValidUnitPeriods,
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

def create_mock_waveforms(nwbfile: NWBFile, num_units: int = 3, num_samples: int = 60,
                          spikes_per_unit: list = None, channels_per_unit: list = None,
                          peak_sample_index: int = 20, random_spikes: RandomSpikes = None):
    """Create mock Waveforms with double-ragged array structure.

    Each unit can have a different number of spikes and each unit can have
    different channels, making the data doubly ragged.
    """
    if spikes_per_unit is None:
        spikes_per_unit = [3, 2, 4]
    if channels_per_unit is None:
        channels_per_unit = [[0, 1, 2], [1, 2, 3, 4], [2, 3]]
    if random_spikes is None:
        random_spikes = create_mock_random_spikes(num_units=num_units, spikes_per_unit=spikes_per_unit)

    all_data_rows = []
    all_electrode_indices = []
    spike_cumulative = []  # data_index: cumulative row count per spike
    unit_cumulative = []   # data_index_index: cumulative spike count per unit

    total_spikes = 0
    running_row_count = 0
    for unit_idx, (n_spikes, unit_channels) in enumerate(zip(spikes_per_unit, channels_per_unit)):
        n_channels = len(unit_channels)
        for spike_idx in range(n_spikes):
            waveform = np.random.randn(n_channels, num_samples).astype(np.float32)
            all_data_rows.append(waveform)
            all_electrode_indices.extend(unit_channels)
            running_row_count += n_channels
            spike_cumulative.append(running_row_count)
        total_spikes += n_spikes
        unit_cumulative.append(total_spikes)

    data_array = np.vstack(all_data_rows).astype(np.float32)

    data = VectorData(
        name="data",
        data=data_array,
        description="Waveform data (one row per channel per spike)",
    )

    data_index = VectorIndex(
        name="data_index",
        data=np.array(spike_cumulative, dtype=np.int64),
        target=data,
    )

    data_index_index = VectorIndex(
        name="data_index_index",
        data=np.array(unit_cumulative, dtype=np.int64),
        target=data_index,
    )

    electrodes = DynamicTableRegion(
        name="electrodes",
        data=list(int(i) for i in all_electrode_indices),
        description="Electrode for each waveform row.",
        table=nwbfile.electrodes,
    )

    waveforms = Waveforms(
        name="waveforms",
        peak_sample_index=peak_sample_index,
        data=data,
        data_index=data_index,
        data_index_index=data_index_index,
        electrodes=electrodes,
        random_spikes=random_spikes,
    )
    return waveforms


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


def create_mock_pca_projections_by_channel(nwbfile: NWBFile, num_units: int = 3, num_components: int = 5,
                                     spikes_per_unit: list = None, channels_per_unit: list = None,
                                     waveforms: Waveforms = None):
    """Create mock PCAProjectionsByChannel with double-ragged array structure."""
    if spikes_per_unit is None:
        spikes_per_unit = [3, 2, 4]
    if channels_per_unit is None:
        channels_per_unit = [[0, 1, 2], [1, 2, 3, 4], [2, 3]]
    if waveforms is None:
        waveforms = create_mock_waveforms(
            nwbfile, num_units=num_units, spikes_per_unit=spikes_per_unit,
            channels_per_unit=channels_per_unit,
        )

    all_data_rows = []
    all_electrode_indices = []
    spike_cumulative = []
    unit_cumulative = []

    total_spikes = 0
    running_row_count = 0
    for unit_idx, (n_spikes, unit_channels) in enumerate(zip(spikes_per_unit, channels_per_unit)):
        n_channels = len(unit_channels)
        for spike_idx in range(n_spikes):
            projections = np.random.randn(n_channels, num_components).astype(np.float64)
            all_data_rows.append(projections)
            all_electrode_indices.extend(unit_channels)
            running_row_count += n_channels
            spike_cumulative.append(running_row_count)
        total_spikes += n_spikes
        unit_cumulative.append(total_spikes)

    data_array = np.vstack(all_data_rows)

    data = VectorData(
        name="data",
        data=data_array,
        description="PCA projections (one row per channel per spike)",
    )

    data_index = VectorIndex(
        name="data_index",
        data=np.array(spike_cumulative, dtype=np.int64),
        target=data,
    )

    data_index_index = VectorIndex(
        name="data_index_index",
        data=np.array(unit_cumulative, dtype=np.int64),
        target=data_index,
    )

    electrodes = DynamicTableRegion(
        name="electrodes",
        data=list(int(i) for i in all_electrode_indices),
        description="Electrode for each projection row.",
        table=nwbfile.electrodes,
    )

    principal_components = PCAProjectionsByChannel(
        name="pca_projections_by_channel",
        data=data,
        data_index=data_index,
        data_index_index=data_index_index,
        electrodes=electrodes,
        waveforms=waveforms,
    )
    return principal_components


def create_mock_pca_projections_concatenated(nwbfile: NWBFile = None, num_units: int = 3, num_components: int = 5,
                                                   spikes_per_unit: list = None,
                                                   waveforms: Waveforms = None):
    """Create mock PCAProjectionsChannelsConcatenated (no electrodes)."""
    if spikes_per_unit is None:
        spikes_per_unit = [3, 2, 4]
    if waveforms is None and nwbfile is not None:
        waveforms = create_mock_waveforms(nwbfile, num_units=num_units, spikes_per_unit=spikes_per_unit)

    all_data_rows = []
    unit_cumulative = []

    total_spikes = 0
    for n_spikes in spikes_per_unit:
        unit_data = np.random.randn(n_spikes, num_components).astype(np.float64)
        all_data_rows.append(unit_data)
        total_spikes += n_spikes
        unit_cumulative.append(total_spikes)

    data_array = np.vstack(all_data_rows)

    data = VectorData(
        name="data",
        data=data_array,
        description="PCA projections (concatenated channels, one row per spike)",
    )

    data_index = VectorIndex(
        name="data_index",
        data=np.array(unit_cumulative, dtype=np.int64),
        target=data,
    )

    pca_projections = PCAProjectionsConcatenated(
        name="pca_projections_concatenated",
        data=data,
        data_index=data_index,
        waveforms=waveforms,
    )
    return pca_projections


class TestRandomSpikesConstructor(TestCase):
    """Unit tests for RandomSpikes constructor."""

    def test_constructor(self):
        """Test that RandomSpikes constructor sets values correctly."""
        random_spikes = create_mock_random_spikes()

        self.assertEqual(random_spikes.name, "random_spikes")
        self.assertEqual(len(random_spikes.random_spikes_indices.data), 33)  # 10 + 15 + 8


class TestWaveformsConstructor(TestCase):
    """Unit tests for Waveforms constructor (double-ragged)."""

    def test_constructor(self):
        """Test that Waveforms constructor sets values correctly."""
        nwbfile = set_up_nwbfile()
        waveforms = create_mock_waveforms(nwbfile)

        self.assertEqual(waveforms.name, "waveforms")
        # spikes_per_unit=[3,2,4] => 9 total spikes
        self.assertEqual(len(waveforms.data_index.data), 9)
        # unit_cumulative => [3, 5, 9]
        self.assertEqual(len(waveforms.data_index_index.data), 3)
        # channels_per_unit=[3,4,2], spikes=[3,2,4]
        # total rows = 3*3 + 2*4 + 4*2 = 9+8+8 = 25
        self.assertEqual(waveforms.data.data.shape[0], 25)
        self.assertEqual(waveforms.data.data.shape[1], 60)
        self.assertEqual(len(waveforms.electrodes.data), 25)

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


class TestPCAProjectionsByChannelConstructor(TestCase):
    """Unit tests for PCAProjectionsByChannel constructor (double-ragged)."""

    def test_constructor(self):
        """Test that PCAProjectionsByChannel constructor sets values correctly."""
        nwbfile = set_up_nwbfile()
        pc = create_mock_pca_projections_by_channel(nwbfile)

        self.assertEqual(pc.name, "pca_projections_by_channel")
        # spikes_per_unit=[3,2,4] => 9 total spikes
        self.assertEqual(len(pc.data_index.data), 9)
        # unit_cumulative => [3, 5, 9]
        self.assertEqual(len(pc.data_index_index.data), 3)
        # channels_per_unit=[3,4,2], spikes=[3,2,4]
        # total rows = 3*3 + 2*4 + 4*2 = 9+8+8 = 25
        self.assertEqual(pc.data.data.shape[0], 25)
        self.assertEqual(pc.data.data.shape[1], 5)  # num_components
        self.assertEqual(len(pc.electrodes.data), 25)


class TestPCAProjectionsConcatenatedConstructor(TestCase):
    """Unit tests for PCAProjectionsChannelsConcatenated constructor."""

    def test_constructor(self):
        """Test PCAProjectionsChannelsConcatenated sets values correctly."""
        nwbfile = set_up_nwbfile()
        pc = create_mock_pca_projections_concatenated(nwbfile)

        self.assertEqual(pc.name, "pca_projections_concatenated")
        # 9 total spikes
        self.assertEqual(pc.data.data.shape, (9, 5))
        # 3 units
        self.assertEqual(len(pc.data_index.data), 3)


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


class TestWaveformsRoundtrip(TestCase):
    """Roundtrip test for Waveforms (double-ragged)."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_waveforms.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading Waveforms with double-ragged array."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        random_spikes = create_mock_random_spikes()
        waveforms = create_mock_waveforms(self.nwbfile, random_spikes=random_spikes)

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.random_spikes = random_spikes
        extensions.waveforms = waveforms

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
            read_waveforms = read_extensions.waveforms

            np.testing.assert_array_almost_equal(
                read_waveforms.data.data[:],
                waveforms.data.data[:],
            )
            np.testing.assert_array_equal(
                read_waveforms.data_index.data[:],
                waveforms.data_index.data[:],
            )
            np.testing.assert_array_equal(
                read_waveforms.data_index_index.data[:],
                waveforms.data_index_index.data[:],
            )
            np.testing.assert_array_equal(
                read_waveforms.electrodes.data[:],
                waveforms.electrodes.data[:],
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


def create_mock_unit_metrics(nwbfile: NWBFile, num_units: int = 3, with_time_support: bool = True):
    """Create a mock UnitMetrics with MetricVectorData columns.

    When ``with_time_support`` is True, a ValidUnitPeriods table is also built
    and each metric column gets its time_support attribute set to reference it.
    Returns (unit_metrics, valid_unit_periods); the second value is None when
    ``with_time_support`` is False.
    """
    from ndx_spikesorting import MetricVectorData

    unit_column = DynamicTableRegion(
        name="unit",
        data=list(range(num_units)),
        description="Reference to nwbfile.units row this metric value applies to.",
        table=nwbfile.units,
    )

    vup = None
    if with_time_support:
        vup = create_mock_valid_unit_periods(nwbfile, num_units=num_units, periods_per_unit=2)

    presence_data = np.random.rand(num_units)
    isi_data = np.random.rand(num_units) * 0.1
    cutoff_data = np.random.rand(num_units) * 0.2

    def _col(name, description, data):
        kwargs = dict(name=name, description=description, data=data)
        if vup is not None:
            kwargs["time_support"] = vup
        return MetricVectorData(**kwargs)

    columns = [
        unit_column,
        _col("presence_ratio", "Fraction of bins in which the unit fired.", presence_data),
        _col("isi_violations_ratio", "Fraction of ISIs below the refractory threshold.", isi_data),
        _col("amplitude_cutoff", "Estimated fraction of spikes missed.", cutoff_data),
    ]

    unit_metrics = UnitMetrics(
        name="quality_metrics",
        description="Run-dependent per-unit metrics from one analysis run.",
        columns=columns,
    )
    return unit_metrics, vup


def create_mock_valid_unit_periods(nwbfile: NWBFile, num_units: int = 3, periods_per_unit: int = 2):
    """Create mock ValidUnitPeriods with time intervals for each unit."""
    start_times = []
    stop_times = []
    unit_indices = []

    for unit_index in range(num_units):
        for p in range(periods_per_unit):
            start = unit_index * 10.0 + p * 3.0
            stop = start + 2.0
            start_times.append(start)
            stop_times.append(stop)
            unit_indices.append(unit_index)

    units = DynamicTableRegion(
        name="unit",
        data=unit_indices,
        description="Reference to units table for each valid period.",
        table=nwbfile.units,
    )

    valid_unit_periods = ValidUnitPeriods(
        name="valid_unit_periods",
        description="Valid periods for each unit.",
        columns=[
            VectorData(name="start_time", data=start_times, description="Start time of each valid period."),
            VectorData(name="stop_time", data=stop_times, description="Stop time of each valid period."),
            units,
        ],
    )
    return valid_unit_periods


class TestUnitMetricsConstructor(TestCase):
    """Unit tests for UnitMetrics constructor."""

    def test_constructor_without_intervals(self):
        """UnitMetrics columns carry no time_support attribute when not requested."""
        nwbfile = set_up_nwbfile()
        run, vup = create_mock_unit_metrics(nwbfile, with_time_support=False)

        self.assertEqual(run.name, "quality_metrics")
        self.assertIn("presence_ratio", run.colnames)
        self.assertIn("isi_violations_ratio", run.colnames)
        self.assertIn("unit", run.colnames)
        self.assertIsNone(vup)
        self.assertIsNone(getattr(run["presence_ratio"], "time_support", None))
        self.assertEqual(len(run["presence_ratio"].data), 3)
        self.assertEqual(len(run["isi_violations_ratio"].data), 3)
        self.assertEqual(len(run["amplitude_cutoff"].data), 3)

    def test_constructor_with_intervals(self):
        """Each metric column references the same ValidUnitPeriods via time_support."""
        nwbfile = set_up_nwbfile()
        run, vup = create_mock_unit_metrics(nwbfile, with_time_support=True)

        self.assertIsNotNone(vup)
        for col_name in ("presence_ratio", "isi_violations_ratio", "amplitude_cutoff"):
            self.assertIs(run[col_name].time_support, vup)


class TestUnitMetricsRoundtrip(TestCase):
    """Roundtrip test for UnitMetrics."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_unit_metrics.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading a UnitMetrics instance with linked ValidUnitPeriods."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        run, vup = create_mock_unit_metrics(self.nwbfile, with_time_support=True)

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.valid_unit_periods = vup
        extensions.add_unit_metrics(run)

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
            read_run = read_extensions.unit_metrics["quality_metrics"]

            np.testing.assert_array_almost_equal(
                read_run["presence_ratio"][:], run["presence_ratio"].data
            )
            np.testing.assert_array_almost_equal(
                read_run["isi_violations_ratio"][:], run["isi_violations_ratio"].data
            )
            # The link survives the round-trip: each metric column resolves to
            # the same ValidUnitPeriods table.
            ts_link = read_run["presence_ratio"].time_support
            self.assertIsNotNone(ts_link)
            np.testing.assert_array_equal(
                np.asarray(ts_link["start_time"][:]),
                np.asarray(vup["start_time"].data),
            )


class TestValidUnitPeriodsConstructor(TestCase):
    """Unit tests for ValidUnitPeriods constructor."""

    def test_constructor(self):
        """Test that ValidUnitPeriods constructor sets values correctly."""
        nwbfile = set_up_nwbfile()
        valid_periods = create_mock_valid_unit_periods(nwbfile)

        self.assertEqual(valid_periods.name, "valid_unit_periods")
        # 3 units * 2 periods each = 6 rows
        self.assertEqual(len(valid_periods["start_time"].data), 6)
        self.assertEqual(len(valid_periods["stop_time"].data), 6)
        self.assertEqual(len(valid_periods["unit"].data), 6)


class TestValidUnitPeriodsRoundtrip(TestCase):
    """Roundtrip test for ValidUnitPeriods."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_valid_unit_periods.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading ValidUnitPeriods."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        valid_periods = create_mock_valid_unit_periods(self.nwbfile)

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.valid_unit_periods = valid_periods

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
            read_valid_periods = read_extensions.valid_unit_periods

            np.testing.assert_array_almost_equal(
                read_valid_periods["start_time"][:],
                valid_periods["start_time"].data,
            )
            np.testing.assert_array_almost_equal(
                read_valid_periods["stop_time"][:],
                valid_periods["stop_time"].data,
            )
            np.testing.assert_array_equal(
                read_valid_periods["unit"].data[:],
                valid_periods["unit"].data,
            )


class TestPCAProjectionsByChannelRoundtrip(TestCase):
    """Roundtrip test for PCAProjectionsByChannel (double-ragged)."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_pca_by_channel.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test writing and reading PCAProjectionsByChannel with double-ragged array."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        random_spikes = create_mock_random_spikes()
        waveforms = create_mock_waveforms(self.nwbfile, random_spikes=random_spikes)
        pc = create_mock_pca_projections_by_channel(self.nwbfile, waveforms=waveforms)

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.random_spikes = random_spikes
        extensions.waveforms = waveforms
        extensions.pca_projections_by_channel = pc

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
            read_pc = read_extensions.pca_projections_by_channel

            np.testing.assert_array_almost_equal(
                read_pc.data.data[:],
                pc.data.data[:],
            )
            np.testing.assert_array_equal(
                read_pc.data_index.data[:],
                pc.data_index.data[:],
            )
            np.testing.assert_array_equal(
                read_pc.data_index_index.data[:],
                pc.data_index_index.data[:],
            )
            np.testing.assert_array_equal(
                read_pc.electrodes.data[:],
                pc.electrodes.data[:],
            )


class TestPCAProjectionsConcatenatedRoundtrip(TestCase):
    """Roundtrip test for PCAProjectionsChannelsConcatenated."""

    def setUp(self):
        self.nwbfile = set_up_nwbfile()
        self.path = "test_pca_concatenated.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_roundtrip(self):
        """Test roundtrip for PCAProjectionsChannelsConcatenated."""
        electrodes_region = self.nwbfile.create_electrode_table_region(
            region=list(range(10)),
            description="all electrodes",
        )
        units_region = create_units_region(self.nwbfile)

        random_spikes = create_mock_random_spikes()
        waveforms = create_mock_waveforms(self.nwbfile, random_spikes=random_spikes)
        pc = create_mock_pca_projections_concatenated(self.nwbfile, waveforms=waveforms)

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.random_spikes = random_spikes
        extensions.waveforms = waveforms
        extensions.pca_projections_concatenated = pc

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
            read_pc = read_extensions.pca_projections_concatenated

            np.testing.assert_array_almost_equal(
                read_pc.data.data[:],
                pc.data.data[:],
            )
            np.testing.assert_array_equal(
                read_pc.data_index.data[:],
                pc.data_index.data[:],
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
        waveforms = create_mock_waveforms(self.nwbfile, random_spikes=random_spikes)
        pca = create_mock_pca_projections_by_channel(self.nwbfile, waveforms=waveforms)

        num_units = 3
        num_channels = 10
        sparsity_mask = np.zeros((num_units, num_channels), dtype=bool)
        sparsity_mask[0, [0, 1, 2]] = True
        sparsity_mask[1, [1, 2, 3, 4]] = True
        sparsity_mask[2, [2, 3]] = True

        extensions = SpikeSortingExtensions(name="extensions")
        extensions.random_spikes = random_spikes
        extensions.waveforms = waveforms
        extensions.templates = templates
        extensions.noise_levels = noise_levels
        extensions.pca_projections_by_channel = pca
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

            read_waveforms = read_extensions.waveforms
            self.assertIsNotNone(read_waveforms)
            self.assertEqual(read_waveforms.peak_sample_index, 20)
            np.testing.assert_array_almost_equal(
                read_waveforms.data.data[:],
                waveforms.data.data[:],
            )

            read_pca = read_extensions.pca_projections_by_channel
            self.assertIsNotNone(read_pca)
            np.testing.assert_array_almost_equal(
                read_pca.data.data[:],
                pca.data.data[:],
            )


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


# ---------------------------------------------------------------------------
# Integration tests for add_sorting_analyzer_to_nwbfile / read_sorting_analyzer_from_nwb
# ---------------------------------------------------------------------------


def _create_sorting_analyzer(compute_all: bool = True):
    """Create a SortingAnalyzer with mock data and optionally compute all extensions."""
    from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording

    recording, sorting = generate_ground_truth_recording(
        durations=[5.0],
        num_units=5,
        num_channels=10,
        seed=42,
    )
    sorting_analyzer = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="memory",
        sparse=True,
    )
    if compute_all:
        sorting_analyzer.compute(
            {
                "random_spikes": {"max_spikes_per_unit": 10, "seed": 42},
                "waveforms": {},
                "templates": {},
                "noise_levels": {},
                "unit_locations": {"method": "monopolar_triangulation"},
                "correlograms": {},
                "principal_components": {"n_components": 3},
                "isi_histograms": {},
                "template_similarity": {},
                "spike_amplitudes": {},
                "amplitude_scalings": {},
                "spike_locations": {"method": "grid_convolution"},
                "quality_metrics": {},
                "template_metrics": {},
                "spiketrain_metrics": {},
            }
        )
    return sorting_analyzer, recording, sorting


def _create_nwbfile_with_recording_and_sorting(recording, sorting):
    """Create an NWBFile populated with recording and sorting via neuroconv."""
    from datetime import datetime, timezone

    from neuroconv.tools.spikeinterface import add_recording_to_nwbfile, add_sorting_to_nwbfile

    nwbfile = NWBFile(
        session_description="Test",
        identifier=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        session_start_time=datetime.now(timezone.utc),
    )
    add_recording_to_nwbfile(recording, nwbfile=nwbfile, write_as="raw", iterator_type=None)
    add_sorting_to_nwbfile(sorting, nwbfile=nwbfile, write_as="units")
    return nwbfile


class TestAddSortingAnalyzerToNwbfile(TestCase):
    """Tests for add_sorting_analyzer_to_nwbfile."""

    def setUp(self):
        from ndx_spikesorting import add_sorting_analyzer_to_nwbfile

        self.add_fn = add_sorting_analyzer_to_nwbfile
        self.sorting_analyzer, self.recording, self.sorting = _create_sorting_analyzer()
        self.nwbfile = _create_nwbfile_with_recording_and_sorting(self.recording, self.sorting)
        self.path = "test_add_sorting_analyzer.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def test_adds_container_to_ecephys(self):
        """Container is placed in processing['ecephys']['spike_sorting']."""
        self.add_fn(self.sorting_analyzer, self.nwbfile)

        self.assertIn("ecephys", self.nwbfile.processing)
        container = self.nwbfile.processing["ecephys"]["spike_sorting"]
        self.assertEqual(container.sampling_frequency, self.sorting_analyzer.sampling_frequency)

    def test_all_extensions_present(self):
        """All computed extensions appear on the NWB container."""
        self.add_fn(self.sorting_analyzer, self.nwbfile)
        ext = self.nwbfile.processing["ecephys"]["spike_sorting"].spike_sorting_extensions

        self.assertIsNotNone(ext.random_spikes)
        self.assertIsNotNone(ext.waveforms)
        self.assertIsNotNone(ext.templates)
        self.assertIsNotNone(ext.noise_levels)
        self.assertIsNotNone(ext.unit_locations)
        self.assertIsNotNone(ext.correlograms)
        self.assertIsNotNone(ext.isi_histograms)
        self.assertIsNotNone(ext.template_similarity)
        self.assertIsNotNone(ext.spike_amplitudes)
        self.assertIsNotNone(ext.spike_locations)
        self.assertIsNotNone(ext.amplitude_scalings)
        self.assertIsNotNone(ext.pca_projections_by_channel)

    def test_sparsity_mask_stored(self):
        """Sparsity mask is written when the analyzer has sparsity."""
        self.add_fn(self.sorting_analyzer, self.nwbfile)
        container = self.nwbfile.processing["ecephys"]["spike_sorting"]

        expected_mask = self.sorting_analyzer.sparsity.mask
        np.testing.assert_array_equal(container.sparsity_mask, expected_mask)

    def test_electrical_series_linked(self):
        """source_electrical_series is auto-detected when exactly one exists."""
        self.add_fn(self.sorting_analyzer, self.nwbfile)
        container = self.nwbfile.processing["ecephys"]["spike_sorting"]

        self.assertIsNotNone(container.source_electrical_series)

    def test_writes_to_disk(self):
        """File written with add_sorting_analyzer_to_nwbfile can be re-read."""
        self.add_fn(self.sorting_analyzer, self.nwbfile)

        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(self.nwbfile)

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            read_nwb = io.read()
            container = read_nwb.processing["ecephys"]["spike_sorting"]
            self.assertEqual(container.sampling_frequency, self.sorting_analyzer.sampling_frequency)
            ext = container.spike_sorting_extensions
            self.assertIsNotNone(ext.templates)

    def test_only_computed_extensions_added(self):
        """When only a subset of extensions is computed, only those are stored."""
        sa, rec, sort = _create_sorting_analyzer(compute_all=False)
        sa.compute({"random_spikes": {"max_spikes_per_unit": 10, "seed": 42}, "templates": {}})

        nwbfile = _create_nwbfile_with_recording_and_sorting(rec, sort)
        self.add_fn(sa, nwbfile)
        ext = nwbfile.processing["ecephys"]["spike_sorting"].spike_sorting_extensions

        self.assertIsNotNone(ext.random_spikes)
        self.assertIsNotNone(ext.templates)
        self.assertIsNone(ext.noise_levels)
        self.assertIsNone(ext.correlograms)

    def test_error_no_units(self):
        """ValueError raised when nwbfile has no units table."""
        from datetime import datetime, timezone

        from neuroconv.tools.spikeinterface import add_recording_to_nwbfile

        nwbfile = NWBFile(
            session_description="Test",
            identifier="no_units",
            session_start_time=datetime.now(timezone.utc),
        )
        add_recording_to_nwbfile(self.recording, nwbfile=nwbfile, write_as="raw", iterator_type=None)

        with self.assertRaises(ValueError):
            self.add_fn(self.sorting_analyzer, nwbfile)

    def test_reuses_existing_ecephys_module(self):
        """If 'ecephys' processing module already exists, it is reused."""
        self.nwbfile.create_processing_module(name="ecephys", description="pre-existing")
        self.add_fn(self.sorting_analyzer, self.nwbfile)

        self.assertEqual(self.nwbfile.processing["ecephys"].description, "pre-existing")
        self.assertIn("spike_sorting", self.nwbfile.processing["ecephys"].data_interfaces)


class TestReadSortingAnalyzerFromNwb(TestCase):
    """Tests for read_sorting_analyzer_from_nwb."""

    def setUp(self):
        from ndx_spikesorting import add_sorting_analyzer_to_nwbfile, read_sorting_analyzer_from_nwb

        self.read_fn = read_sorting_analyzer_from_nwb
        self.sorting_analyzer, self.recording, self.sorting = _create_sorting_analyzer()
        nwbfile = _create_nwbfile_with_recording_and_sorting(self.recording, self.sorting)
        add_sorting_analyzer_to_nwbfile(self.sorting_analyzer, nwbfile)

        self.path = "test_read_sorting_analyzer.nwb"
        with NWBHDF5IO(self.path, mode="w") as io:
            io.write(nwbfile)

    def tearDown(self):
        del self.sorting_analyzer, self.recording, self.sorting
        remove_test_file(self.path)

    def test_loads_all_extensions(self):
        """Extensions restored from the NWB file.

        Quality, template, and spiketrain metric columns all live on a
        single UnitMetrics table on disk; the reader uses
        ``COLUMN_TO_EXTENSION`` to split them back into the right SI
        extensions on read. ``FiringRate`` is the only canonized typed
        column on Units in v1 (additional canonical columns are in
        follow-up PRs); it routes to ``spiketrain_metrics``.
        """
        sa = self.read_fn(self.path)
        expected = {
            "random_spikes", "waveforms", "templates", "noise_levels",
            "unit_locations", "correlograms", "isi_histograms",
            "template_similarity", "spike_amplitudes", "amplitude_scalings",
            "spike_locations", "principal_components",
            "quality_metrics", "template_metrics", "spiketrain_metrics",
        }
        self.assertEqual(set(sa.extensions.keys()), expected)

    def test_unit_ids_match(self):
        """Unit IDs from the read analyzer match the original sorting."""
        sa = self.read_fn(self.path)
        # NwbSortingExtractor returns string IDs
        original_ids = [str(uid) for uid in self.sorting_analyzer.unit_ids]
        np.testing.assert_array_equal(sa.unit_ids, original_ids)

    def test_num_channels(self):
        """Number of channels matches the original recording."""
        sa = self.read_fn(self.path)
        self.assertEqual(sa.get_num_channels(), self.sorting_analyzer.get_num_channels())

    def test_sampling_frequency(self):
        """Sampling frequency matches the original."""
        sa = self.read_fn(self.path)
        self.assertEqual(sa.sampling_frequency, self.sorting_analyzer.sampling_frequency)

    def test_sparsity_restored(self):
        """Sparsity mask is correctly restored."""
        sa = self.read_fn(self.path)
        self.assertIsNotNone(sa.sparsity)
        np.testing.assert_array_equal(sa.sparsity.mask, self.sorting_analyzer.sparsity.mask)

    def test_templates_shape(self):
        """Template shapes match (num_units, num_samples, num_channels)."""
        sa = self.read_fn(self.path)
        original = self.sorting_analyzer.get_extension("templates").data["average"]
        restored = sa.get_extension("templates").data["average"]
        self.assertEqual(restored.shape, original.shape)

    def test_templates_values(self):
        """Template values are close to the original after roundtrip."""
        sa = self.read_fn(self.path)
        original = self.sorting_analyzer.get_extension("templates").data["average"]
        restored = sa.get_extension("templates").data["average"]
        np.testing.assert_array_almost_equal(restored, original, decimal=5)

    def test_random_spikes_count(self):
        """Number of random spikes matches."""
        sa = self.read_fn(self.path)
        original_count = len(self.sorting_analyzer.get_extension("random_spikes").get_random_spikes())
        restored_count = len(sa.get_extension("random_spikes").get_random_spikes())
        self.assertEqual(restored_count, original_count)

    def test_waveforms_shape(self):
        """Waveform data shape is preserved."""
        sa = self.read_fn(self.path)
        restored_wf = sa.get_extension("waveforms").data["waveforms"]
        original_wf = self.sorting_analyzer.get_extension("waveforms").data["waveforms"]
        self.assertEqual(restored_wf.shape[0], original_wf.shape[0])  # num spikes
        self.assertEqual(restored_wf.shape[1], original_wf.shape[1])  # num samples

    def test_noise_levels(self):
        """Noise levels values match."""
        sa = self.read_fn(self.path)
        original = self.sorting_analyzer.get_extension("noise_levels").data["noise_levels"]
        restored = sa.get_extension("noise_levels").data["noise_levels"]
        np.testing.assert_array_almost_equal(restored, original, decimal=5)

    def test_correlograms_shape(self):
        """Correlogram data shape is preserved."""
        sa = self.read_fn(self.path)
        original = self.sorting_analyzer.get_extension("correlograms").data["ccgs"]
        restored = sa.get_extension("correlograms").data["ccgs"]
        self.assertEqual(restored.shape, original.shape)

    def test_pca_projections_shape(self):
        """PCA projection shape is preserved."""
        sa = self.read_fn(self.path)
        restored_pca = sa.get_extension("principal_components").data["pca_projection"]
        original_pca = self.sorting_analyzer.get_extension("principal_components").data["pca_projection"]
        self.assertEqual(restored_pca.shape[0], original_pca.shape[0])  # num spikes
        self.assertEqual(restored_pca.shape[1], original_pca.shape[1])  # num components

    def test_custom_container_path(self):
        """read_sorting_analyzer_from_nwb works with explicit container_path."""
        sa = self.read_fn(self.path, container_path="ecephys/spike_sorting")
        self.assertIsNotNone(sa)
        self.assertIn("templates", sa.extensions)

    def test_cell_intrinsic_metrics_on_units(self):
        """Cell-intrinsic metric columns land on nwbfile.units with type tags."""
        from pynwb import NWBHDF5IO
        from ndx_spikesorting import FiringRate

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            nwbfile = io.read()
            self.assertIsNotNone(nwbfile.units)
            # FiringRate is the only canonical typed column in v1.
            typed_columns = {
                col_name: type(nwbfile.units[col_name]).__name__
                for col_name in nwbfile.units.colnames
                if isinstance(nwbfile.units[col_name], FiringRate)
            }
            self.assertGreater(len(typed_columns), 0)

    def test_unit_metrics_present(self):
        """Run-dependent metrics land in a UnitMetrics instance inside extensions."""
        from pynwb import NWBHDF5IO

        with NWBHDF5IO(self.path, mode="r", load_namespaces=True) as io:
            nwbfile = io.read()
            ext = nwbfile.processing["ecephys"]["spike_sorting"].spike_sorting_extensions
            self.assertIn("quality_metrics", ext.unit_metrics)

    def test_metric_values_roundtrip(self):
        """Reconstructed SI metric extensions hold the same values."""
        sa = self.read_fn(self.path)

        original_st = self.sorting_analyzer.get_extension("spiketrain_metrics").get_data()
        restored_st = sa.get_extension("spiketrain_metrics").get_data()
        np.testing.assert_array_almost_equal(
            restored_st["firing_rate"].to_numpy(),
            original_st["firing_rate"].to_numpy(),
        )

        original_qm = self.sorting_analyzer.get_extension("quality_metrics").get_data()
        restored_qm = sa.get_extension("quality_metrics").get_data()
        for col in ("presence_ratio", "isi_violations_ratio"):
            # NaN-safe comparison.
            np.testing.assert_allclose(
                restored_qm[col].to_numpy(),
                original_qm[col].to_numpy(),
                equal_nan=True,
            )

    def test_template_metrics_roundtrip(self):
        """template_metrics columns split out of UnitMetrics into the right SI extension."""
        sa = self.read_fn(self.path)

        original_tm = self.sorting_analyzer.get_extension("template_metrics").get_data()
        restored_tm = sa.get_extension("template_metrics").get_data()
        # Sanity check: the SI extension is restored, not merged into quality_metrics.
        self.assertIsNotNone(restored_tm)
        # Every template-metric column from the original is present and equal.
        for col in original_tm.columns:
            self.assertIn(col, restored_tm.columns, f"template_metrics missing {col!r}")
            np.testing.assert_allclose(
                restored_tm[col].to_numpy(),
                original_tm[col].to_numpy(),
                equal_nan=True,
            )

    def test_quality_and_template_metrics_separated_on_read(self):
        """quality_metrics and template_metrics extensions hold disjoint columns post-read.

        Pre-fix, every column on UnitMetrics was funneled into quality_metrics,
        which made template_metrics empty (or absent) and put template columns
        like ``peak_to_trough_duration`` under quality_metrics.
        """
        sa = self.read_fn(self.path)

        qm_cols = set(sa.get_extension("quality_metrics").get_data().columns)
        tm_cols = set(sa.get_extension("template_metrics").get_data().columns)

        # template_metrics columns should NOT be in quality_metrics.
        leaked = {"peak_to_trough_duration", "trough_half_width", "peak_half_width"} & qm_cols
        self.assertEqual(leaked, set(), f"template columns leaked into quality_metrics: {leaked}")

        # quality_metrics columns should NOT be in template_metrics.
        leaked = {"presence_ratio", "isi_violations_ratio"} & tm_cols
        self.assertEqual(leaked, set(), f"quality columns leaked into template_metrics: {leaked}")

    def test_column_to_extension_covers_all_computed_columns(self):
        """COLUMN_TO_EXTENSION covers every column produced by the SI extensions
        we compute in the fixture.

        Acts as a guard: if SI adds a new column in a release, the test fails
        with a clear message naming the uncovered column. Update
        ``COLUMN_TO_EXTENSION`` in ``utils.py`` to silence.
        """
        from ndx_spikesorting.utils import COLUMN_TO_EXTENSION

        uncovered = []
        for ext_name in ("template_metrics", "quality_metrics", "spiketrain_metrics"):
            ext = self.sorting_analyzer.get_extension(ext_name)
            if ext is None:
                continue
            df_cols = ext.get_data().columns
            for col in df_cols:
                if col not in COLUMN_TO_EXTENSION:
                    uncovered.append((ext_name, col))

        self.assertEqual(
            uncovered,
            [],
            "Columns produced by SpikeInterface (SI) but missing from "
            f"COLUMN_TO_EXTENSION: {uncovered}. Add them to the dict in "
            "src/pynwb/ndx_spikesorting/utils.py.",
        )


class TestUnitMetricsTimeSupportLinkRoundtrip(TestCase):
    """Round-trip the time_support column-attribute link through a SortingAnalyzer.

    Exercises three properties that the column-attribute design has to satisfy:
    (1) when quality_metrics is computed with ``periods=<array>``, those periods
    round-trip through a linked ValidUnitPeriods table; (2) the DTR is
    id-aware, so a permuted nwbfile.units does not corrupt unit assignment;
    (3) the absence of qm periods is also handled cleanly.
    """

    def setUp(self):
        from datetime import datetime, timezone

        from spikeinterface.core import create_sorting_analyzer, generate_ground_truth_recording
        from spikeinterface.core.base import unit_period_dtype
        from neuroconv.tools.spikeinterface import add_recording_to_nwbfile, add_sorting_to_nwbfile

        self.unit_period_dtype = unit_period_dtype
        self.add_recording_to_nwbfile = add_recording_to_nwbfile
        self.add_sorting_to_nwbfile = add_sorting_to_nwbfile
        self.datetime, self.timezone = datetime, timezone

        recording, sorting = generate_ground_truth_recording(
            durations=[5.0], num_units=3, num_channels=8, seed=42
        )
        self.recording, self.sorting = recording, sorting
        self.sorting_analyzer = create_sorting_analyzer(
            sorting=sorting, recording=recording, format="memory", sparse=True
        )
        self.sorting_analyzer.compute(
            {"random_spikes": {"max_spikes_per_unit": 10, "seed": 42}, "templates": {}, "noise_levels": {}}
        )

        fs = self.sorting_analyzer.sampling_frequency
        n_units = len(self.sorting_analyzer.unit_ids)
        periods = []
        for u in range(n_units):
            periods.append((0, int(0.0 * fs), int(1.5 * fs), u))
            periods.append((0, int(2.0 * fs), int(4.5 * fs), u))
        self.periods_arr = np.array(periods, dtype=unit_period_dtype)
        self.sorting_analyzer.compute(
            {"quality_metrics": {"periods": self.periods_arr, "metric_names": ["firing_rate", "presence_ratio"]}}
        )

        self.path = "test_unit_metrics_time_support_link.nwb"

    def tearDown(self):
        remove_test_file(self.path)

    def _build_nwbfile(self, units_unit_ids=None):
        nwbfile = NWBFile(
            session_description="t",
            identifier="t",
            session_start_time=self.datetime.now(self.timezone.utc),
        )
        self.add_recording_to_nwbfile(self.recording, nwbfile=nwbfile, write_as="raw", iterator_type=None)
        self.add_sorting_to_nwbfile(self.sorting, nwbfile=nwbfile, write_as="units")
        if units_unit_ids is not None:
            # Permute the unit_name column to simulate a units table whose row
            # order does not match the analyzer's unit_ids order.
            unit_name_col = nwbfile.units["unit_name"]
            unit_name_col.data[:] = units_unit_ids
        return nwbfile

    def test_periods_round_trip_via_link(self):
        """qm periods round-trip to a linked ValidUnitPeriods and back."""
        from ndx_spikesorting import add_sorting_analyzer_to_nwbfile, read_sorting_analyzer_from_nwb

        nwbfile = self._build_nwbfile()
        add_sorting_analyzer_to_nwbfile(self.sorting_analyzer, nwbfile)

        # In-memory: metric columns carry the link and the standalone table exists.
        ext = nwbfile.processing["ecephys"]["spike_sorting"].spike_sorting_extensions
        um = ext.unit_metrics["quality_metrics"]
        self.assertIsNotNone(um["presence_ratio"].time_support)
        self.assertIsNotNone(ext.valid_unit_periods)
        # No legacy inline column.
        self.assertNotIn("time_support", um.colnames)

        with NWBHDF5IO(self.path, "w") as io:
            io.write(nwbfile)
        sa2 = read_sorting_analyzer_from_nwb(self.path)

        p2 = sa2.get_extension("quality_metrics").params.get("periods")
        self.assertIsNotNone(p2)
        orig_sort = np.sort(self.periods_arr, order=["unit_index", "start_sample_index"])
        new_sort = np.sort(p2, order=["unit_index", "start_sample_index"])
        np.testing.assert_array_equal(orig_sort["start_sample_index"], new_sort["start_sample_index"])
        np.testing.assert_array_equal(orig_sort["end_sample_index"], new_sort["end_sample_index"])
        np.testing.assert_array_equal(orig_sort["unit_index"], new_sort["unit_index"])

    def test_periods_round_trip_with_permuted_units(self):
        """A permuted nwbfile.units row order does not corrupt unit assignment.

        After permuting unit_name on the Units table, the id-aware DTR ensures
        each metric value still ends up associated with the correct unit on
        round-trip. The qm periods are reconstructed correctly per unit too.
        """
        from ndx_spikesorting import add_sorting_analyzer_to_nwbfile, read_sorting_analyzer_from_nwb

        analyzer_ids = [str(u) for u in self.sorting_analyzer.unit_ids]
        # Reverse the names so row i carries the unit that the analyzer has at
        # position (n - 1 - i).
        permuted = list(reversed(analyzer_ids))

        nwbfile = self._build_nwbfile(units_unit_ids=permuted)
        add_sorting_analyzer_to_nwbfile(self.sorting_analyzer, nwbfile)

        with NWBHDF5IO(self.path, "w") as io:
            io.write(nwbfile)
        sa2 = read_sorting_analyzer_from_nwb(self.path)

        # Reconstructed analyzer's unit order follows the on-disk units table,
        # which is the permuted order. Metric values per unit must match the
        # original analyzer's, looked up by id rather than position.
        original_qm = self.sorting_analyzer.get_extension("quality_metrics").get_data()
        restored_qm = sa2.get_extension("quality_metrics").get_data()
        for unit_id in original_qm.index:
            uid = str(unit_id)
            self.assertIn(uid, restored_qm.index.astype(str).tolist())
            for col in ("firing_rate", "presence_ratio"):
                if col not in restored_qm.columns:
                    continue
                expected = original_qm.loc[unit_id, col]
                got = restored_qm.loc[uid, col] if uid in restored_qm.index else restored_qm.loc[unit_id, col]
                if not np.isnan(expected):
                    self.assertAlmostEqual(expected, got, places=5)

        # Periods come back with correct unit positions relative to the
        # reconstructed analyzer.
        p2 = sa2.get_extension("quality_metrics").params.get("periods")
        self.assertIsNotNone(p2)
        # Reverse-map: for each restored period, the analyzer-position must
        # round-trip to the same unit id as the original.
        restored_unit_ids_by_pos = list(sa2.unit_ids)
        for p_orig in self.periods_arr:
            orig_uid = analyzer_ids[int(p_orig["unit_index"])]
            # Find any matching period in restored: same start/end, unit id matches.
            mask = (
                (p2["start_sample_index"] == int(p_orig["start_sample_index"]))
                & (p2["end_sample_index"] == int(p_orig["end_sample_index"]))
            )
            candidate_positions = p2["unit_index"][mask]
            candidate_uids = {restored_unit_ids_by_pos[int(pos)] for pos in candidate_positions}
            self.assertIn(orig_uid, candidate_uids)

    def test_no_qm_periods_leaves_no_link(self):
        """When qm has no periods, no ValidUnitPeriods is written and no link is set."""
        from ndx_spikesorting import add_sorting_analyzer_to_nwbfile

        # Re-compute quality_metrics without periods.
        self.sorting_analyzer.compute(
            {"quality_metrics": {"metric_names": ["firing_rate"], "delete_existing_metrics": True}}
        )

        nwbfile = self._build_nwbfile()
        add_sorting_analyzer_to_nwbfile(self.sorting_analyzer, nwbfile)

        ext = nwbfile.processing["ecephys"]["spike_sorting"].spike_sorting_extensions
        um = ext.unit_metrics["quality_metrics"]
        self.assertIsNone(getattr(um["firing_rate"], "time_support", None))
        self.assertIsNone(ext.valid_unit_periods)

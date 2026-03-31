# -*- coding: utf-8 -*-
from pathlib import Path

from pynwb.spec import (
    NWBNamespaceBuilder,
    export_spec,
    NWBGroupSpec,
    NWBAttributeSpec,
    NWBDatasetSpec,
    NWBLinkSpec,
)


def main():
    # these arguments were auto-generated from your cookiecutter inputs
    ns_builder = NWBNamespaceBuilder(
        name="""ndx-spikesorting""",
        version="""0.1.0""",
        doc="""An NWB extension for spike sorting outputs and extensions""",
        author=[
            "Alessio Buccino",
            "Heberto Mayorquin",
        ],
        contact=[
            "h.mayorquin@gmail.com",
        ],
    )
    ns_builder.include_namespace("core")

    # RandomSpikes: stores random spike indices per unit for waveform extraction
    random_spikes = NWBGroupSpec(
        neurodata_type_def="RandomSpikes",
        neurodata_type_inc="NWBDataInterface",
        default_name="random_spikes",
        doc=(
            "Random spike indices per unit for waveform extraction and template computation. "
            "Stores indices into each unit's spike train as a ragged array."
        ),
        datasets=[
            NWBDatasetSpec(
                name="random_spikes_indices",
                neurodata_type_inc="VectorData",
                dtype="int64",
                dims=["num_spikes_total"],
                shape=[None],
                doc=(
                    "Concatenated array of spike indices for all units. Each value is an index "
                    "into the spike train for that unit. Use random_spikes_indices_index to "
                    "retrieve indices for each unit."
                ),
            ),
            NWBDatasetSpec(
                name="random_spikes_indices_index",
                neurodata_type_inc="VectorIndex",
                doc=(
                    "Index into random_spikes_indices for each unit. The indices for unit i are "
                    "random_spikes_indices[random_spikes_indices_index[i-1]:random_spikes_indices_index[i]]."
                ),
            ),
        ],
    )

    # Waveforms: individual spike waveforms as a double-ragged array
    waveforms = NWBGroupSpec(
        neurodata_type_def="Waveforms",
        neurodata_type_inc="NWBDataInterface",
        default_name="waveforms",
        doc=(
            "Individual spike waveforms organized as a double-ragged array. Each row "
            "in data is one channel's waveform for one spike. The first VectorIndex "
            "(data_index) groups rows by spike, and the second VectorIndex "
            "(data_index_index) groups spikes by unit."
        ),
        datasets=[
            NWBDatasetSpec(
                name="data",
                neurodata_type_inc="VectorData",
                dtype="float32",
                dims=["num_channel_waveforms_total", "num_samples"],
                shape=[None, None],
                doc=(
                    "Flattened waveform data. Each row is one channel's waveform for one "
                    "spike. Shape is (total_channel_waveforms, num_samples)."
                ),
                attributes=[
                    NWBAttributeSpec(
                        name="unit",
                        dtype="text",
                        default_value="microvolts",
                        doc="Unit of measurement for waveform data.",
                        required=False,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name="data_index",
                neurodata_type_inc="VectorIndex",
                doc=(
                    "Index into data per spike. Groups channel-waveforms by spike. "
                    "The waveforms for spike j are data[data_index[j-1]:data_index[j], :]."
                ),
            ),
            NWBDatasetSpec(
                name="data_index_index",
                neurodata_type_inc="VectorIndex",
                doc=(
                    "Index into data_index per unit. Groups spikes by unit. "
                    "The spike indices for unit i are "
                    "data_index[data_index_index[i-1]:data_index_index[i]]."
                ),
            ),
            NWBDatasetSpec(
                name="electrodes",
                neurodata_type_inc="DynamicTableRegion",
                doc=(
                    "Reference to the electrodes table for each row in data. Has the "
                    "same length as the first dimension of data, identifying the channel "
                    "for each waveform row."
                ),
            ),
        ],
    )

    # Templates: sparse template waveforms per unit
    templates = NWBGroupSpec(
        neurodata_type_def="Templates",
        neurodata_type_inc="NWBDataInterface",
        default_name="templates",
        doc=(
            "Template waveforms per unit stored as a ragged array. Each row is one "
            "channel's waveform for one unit. When sparsity is used, only active channels "
            "are stored per unit. Without sparsity, all channels are stored for each unit."
        ),
        attributes=[
            NWBAttributeSpec(
                name="peak_sample_index",
                dtype="int32",
                doc=(
                    "The index of the peak sample in the template waveform (0-indexed). "
                    "This is the alignment point used during spike sorting, typically the point "
                    "of maximum absolute amplitude. Combined with the number of samples from the "
                    "data shape, this fully specifies the temporal structure: samples_before = peak_sample_index, "
                    "samples_after = n_samples - peak_sample_index."
                ),
            ),
        ],
        datasets=[
            NWBDatasetSpec(
                name="data",
                neurodata_type_inc="VectorData",
                dtype="float32",
                dims=["num_waveforms", "num_samples"],
                shape=[None, None],
                doc=(
                    "Template waveforms as a ragged array. Shape is "
                    "(total_waveforms, num_samples) where total_waveforms is the sum of "
                    "channels per unit across all units. Use data_index to retrieve "
                    "waveforms for each unit, and electrodes to identify which electrode "
                    "each row corresponds to."
                ),
                attributes=[
                    NWBAttributeSpec(
                        name="unit",
                        dtype="text",
                        default_value="microvolts",
                        doc="Unit of measurement for template data.",
                        required=False,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name="data_index",
                neurodata_type_inc="VectorIndex",
                doc=(
                    "Index into data for each unit. The templates for unit i are "
                    "data[data_index[i-1]:data_index[i], :]."
                ),
            ),
            NWBDatasetSpec(
                name="electrodes",
                neurodata_type_inc="DynamicTableRegion",
                doc=(
                    "Reference to the electrodes table for each row in data. Has the "
                    "same length as the first dimension of data, so data_index indexes both "
                    "datasets in lockstep. The electrodes for unit i are "
                    "electrodes[data_index[i-1]:data_index[i]]."
                ),
            ),
        ],
    )

    # NoiseLevels: noise levels per channel
    noise_levels = NWBGroupSpec(
        neurodata_type_def="NoiseLevels",
        neurodata_type_inc="NWBDataInterface",
        default_name="noise_levels",
        doc="Noise levels estimated after preprocessing for each channel",
        datasets=[
            NWBDatasetSpec(
                name="data",
                dtype="float32",
                dims=["num_channels"],
                shape=[None],
                doc=(
                    "Array of noise levels for each channel. Order and length must match "
                    "SpikeSortingContainer.electrodes."
                ),
                attributes=[
                    NWBAttributeSpec(
                        name="unit",
                        dtype="text",
                        default_value="microvolts",
                        doc="Unit of measurement for noise data.",
                        required=False,
                    ),
                ],
            ),
        ],
    )

    # UnitLocations: estimated locations for each unit
    unit_locations = NWBGroupSpec(
        neurodata_type_def="UnitLocations",
        neurodata_type_inc="NWBDataInterface",
        default_name="unit_locations",
        doc="Estimated locations for each unit in x-y and optionally z dimensions.",
        datasets=[
            NWBDatasetSpec(
                name="data",
                dtype="float",
                dims=[["num_units", "x, y"], ["num_units", "x, y, z"]],
                shape=[[None, 2], [None, 3]],
                doc=(
                    "Locations of units. Shape is (num_units, 2) for 2D locations (x-y) or "
                    "(num_units, 3) for 3D locations (x-y-z)."
                ),
                attributes=[
                    NWBAttributeSpec(
                        name="unit",
                        dtype="text",
                        default_value="micrometers",
                        doc="Unit of measurement for unit locations data.",
                        required=False,
                    ),
                ],
            ),
        ],
    )

    # Correlograms: cross-correlogram spike counts
    correlograms = NWBGroupSpec(
        neurodata_type_def="Correlograms",
        neurodata_type_inc="NWBDataInterface",
        default_name="correlograms",
        doc=(
            "Cross-correlogram spike counts for each unit pair. Diagonal entries are "
            "auto-correlograms. Each value is the number of spike coincidences within "
            "the corresponding time lag bin."
        ),
        datasets=[
            NWBDatasetSpec(
                name="data",
                dtype="int",
                dims=["num_units", "num_units", "num_bins"],
                shape=[None, None, None],
                doc="Correlograms for each pair of unit.",
            ),
            NWBDatasetSpec(
                name="bin_edges",
                dtype="float",
                dims=["num_bin_edges"],
                shape=[None],
                doc="bin edges in ms over which counts are computed.",
            ),
        ],
    )

    # ISIHistograms: inter-spike-interval histogram counts
    isi_histograms = NWBGroupSpec(
        neurodata_type_def="ISIHistograms",
        neurodata_type_inc="NWBDataInterface",
        default_name="isi_histograms",
        doc=(
            "Inter-spike-interval histogram counts for each unit. Each value is the "
            "number of consecutive spike pairs whose interval falls within the "
            "corresponding bin."
        ),
        datasets=[
            NWBDatasetSpec(
                name="data",
                dtype="int",
                dims=["num_units", "num_bins"],
                shape=[None, None],
                doc="ISI histograms for each unit.",
            ),
            NWBDatasetSpec(
                name="bin_edges",
                dtype="float",
                dims=["num_bin_edges"],
                shape=[None],
                doc="bin edges in ms over which counts are computed.",
            ),
        ],
    )

    # TemplateSimilarity: similarity values for each unit pair
    template_similarity = NWBGroupSpec(
        neurodata_type_def="TemplateSimilarity",
        neurodata_type_inc="NWBDataInterface",
        default_name="template_similarity",
        doc="The template similarity for each unit pair.",
        datasets=[
            NWBDatasetSpec(
                name="data",
                dtype="float",
                dims=["num_units", "num_units"],
                shape=[None, None],
                doc="The similarity value for each unit pair.",
            ),
        ],
    )

    # SpikeAmplitudes: amplitude of each spike
    spike_amplitudes = NWBGroupSpec(
        neurodata_type_def="SpikeAmplitudes",
        neurodata_type_inc="NWBDataInterface",
        default_name="spike_amplitudes",
        doc="The amplitude of each spike in uV.",
        datasets=[
            NWBDatasetSpec(
                name="data",
                neurodata_type_inc="VectorData",
                dtype="float",
                dims=["num_spikes_total"],
                shape=[None],
                doc=(
                    "Concatenated array of spike amplitudes for all units. Use spike_amplitude_index "
                    "to retrieve indices for each unit."
                ),
                attributes=[
                    NWBAttributeSpec(
                        name="unit",
                        dtype="text",
                        default_value="microvolts",
                        doc="Unit of measurement for spike amplitudes data.",
                        required=False,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name="data_index",
                neurodata_type_inc="VectorIndex",
                doc=(
                    "Index into data for each unit. The indices for unit i "
                    "are data[data_index[i-1]:data_index[i]]."
                ),
            ),
        ],
    )

    # SpikeLocations: estimated location of each spike
    spike_locations = NWBGroupSpec(
        neurodata_type_def="SpikeLocations",
        neurodata_type_inc="NWBDataInterface",
        default_name="spike_locations",
        doc="The estimated location of each spike in 2D or 3D.",
        datasets=[
            NWBDatasetSpec(
                name="data",
                neurodata_type_inc="VectorData",
                dtype="float",
                dims=[["num_spikes_total", "x, y"], ["num_spikes_total", "x, y, z"]],
                shape=[[None, 2], [None, 3]],
                doc=(
                    "Concatenated array of spike locations for all units. Use spike_location_index "
                    "to retrieve indices for each unit."
                ),
                attributes=[
                    NWBAttributeSpec(
                        name="unit",
                        dtype="text",
                        default_value="micrometers",
                        doc="Unit of measurement for spike locations data.",
                        required=False,
                    ),
                ],
            ),
            NWBDatasetSpec(
                name="data_index",
                neurodata_type_inc="VectorIndex",
                doc=(
                    "Index into data for each unit. The indices for unit i "
                    "are data[data_index[i-1]:data_index[i]]."
                ),
            ),
        ],
    )

    # AmplitudeScalings: relative amplitude of each spike w.r.t. template
    amplitude_scalings = NWBGroupSpec(
        neurodata_type_def="AmplitudeScalings",
        neurodata_type_inc="NWBDataInterface",
        default_name="amplitude_scalings",
        doc=(
            "The amplitude scaling of each spike (the relative amplitude of the spike with "
            "respect to the template)"
        ),
        datasets=[
            NWBDatasetSpec(
                name="data",
                neurodata_type_inc="VectorData",
                dtype="float32",
                dims=["num_spikes_total"],
                shape=[None],
                doc=(
                    "Concatenated array of amplitude scalings for all units. Use amplitude_scalings_index "
                    "to retrieve indices for each unit."
                ),
            ),
            NWBDatasetSpec(
                name="data_index",
                neurodata_type_inc="VectorIndex",
                doc=(
                    "Index into data for each unit. The indices for unit i "
                    "are data[data_index[i-1]:data_index[i]]."
                ),
            ),
        ],
    )

    # PrincipalComponents: PCA projections as a double-ragged array
    principal_components = NWBGroupSpec(
        neurodata_type_def="PrincipalComponents",
        neurodata_type_inc="NWBDataInterface",
        default_name="principal_components",
        doc=(
            "PCA projections of spikes organized as a double-ragged array. Each row "
            "in data is one channel's PCA projection for one spike (per-channel mode) or "
            "the concatenated projection across all channels (concatenated mode). The "
            "first VectorIndex (data_index) groups rows by spike, and the second VectorIndex "
            "(data_index_index) groups spikes by unit. When channels are concatenated "
            "(e.g. tetrodes), each spike produces a single row and electrodes is omitted."
        ),
        datasets=[
            NWBDatasetSpec(
                name="data",
                neurodata_type_inc="VectorData",
                dtype="float",
                dims=["num_channel_projections_total", "num_components"],
                shape=[None, None],
                doc=(
                    "Flattened PCA projection data. Each row is one channel's projection "
                    "for one spike (per-channel mode) or the concatenated projection across "
                    "all channels for one spike (concatenated mode). Shape is "
                    "(total_rows, num_components)."
                ),
            ),
            NWBDatasetSpec(
                name="data_index",
                neurodata_type_inc="VectorIndex",
                doc=(
                    "Index into data per spike. Groups channel-projections by spike. "
                    "The projections for spike j are data[data_index[j-1]:data_index[j], :]."
                ),
            ),
            NWBDatasetSpec(
                name="data_index_index",
                neurodata_type_inc="VectorIndex",
                doc=(
                    "Index into data_index per unit. Groups spikes by unit. "
                    "The spike indices for unit i are "
                    "data_index[data_index_index[i-1]:data_index_index[i]]."
                ),
            ),
            NWBDatasetSpec(
                name="electrodes",
                neurodata_type_inc="DynamicTableRegion",
                quantity="?",
                doc=(
                    "Reference to the electrodes table for each row in data. Has the "
                    "same length as the first dimension of data, identifying the channel "
                    "for each projection row. Omitted when PCA is computed on concatenated channels."
                ),
            ),
        ],
    )

    # SpikeSortingExtensions: container group for all computed extensions
    spike_sorting_extensions = NWBGroupSpec(
        neurodata_type_def="SpikeSortingExtensions",
        neurodata_type_inc="NWBDataInterface",
        default_name="extensions",
        doc=(
            "Container for spike sorting computed extensions such as templates, "
            "random spikes, quality metrics, and other derived data."
        ),
        groups=[
            NWBGroupSpec(
                neurodata_type_inc="RandomSpikes",
                quantity="?",
                doc="Random spikes extension data.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="Waveforms",
                quantity="?",
                doc="Individual spike waveforms (double-ragged).",
            ),
            NWBGroupSpec(
                neurodata_type_inc="Templates",
                quantity="?",
                doc="Templates extension data.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="NoiseLevels",
                quantity="?",
                doc="Noise levels extension data.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="UnitLocations",
                quantity="?",
                doc="Unit locations extension data.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="Correlograms",
                quantity="?",
                doc="Unit correlograms.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="ISIHistograms",
                quantity="?",
                doc="Unit ISI histograms data.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="TemplateSimilarity",
                quantity="?",
                doc="Unit template similarities.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="SpikeAmplitudes",
                quantity="?",
                doc="Spike amplitudes extension data.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="SpikeLocations",
                quantity="?",
                doc="Spike locations extension data.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="AmplitudeScalings",
                quantity="?",
                doc="Amplitudes scalings extension data.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="PrincipalComponents",
                quantity="?",
                doc="PCA projections of spikes.",
            ),
        ],
    )

    # SpikeSortingContainer: main container for spike sorting data
    spike_sorting_container = NWBGroupSpec(
        neurodata_type_def="SpikeSortingContainer",
        neurodata_type_inc="NWBDataInterface",
        default_name="spike_sorting",
        doc=(
            "Container for spike sorting results. Contains metadata about "
            "the sorting analysis, links to electrodes and units tables, and computed extension data."
        ),
        attributes=[
            NWBAttributeSpec(
                name="sampling_frequency",
                dtype="float64",
                doc="Sampling frequency of the recording in Hz.",
            ),
        ],
        datasets=[
            NWBDatasetSpec(
                name="sparsity_mask",
                dtype="bool",
                dims=["num_units", "num_channels"],
                shape=[None, None],
                quantity="?",
                doc=(
                    "Boolean mask indicating which channels are active (sparse) for each unit. "
                    "Shape is (num_units, num_channels). True indicates the channel is used for that unit."
                ),
            ),
            NWBDatasetSpec(
                name="electrodes",
                neurodata_type_inc="DynamicTableRegion",
                doc=(
                    "Reference to the electrodes table rows that correspond to the channels "
                    "used in this sorting analysis. Order matches the channel dimension in templates and sparsity_mask."
                ),
            ),
            NWBDatasetSpec(
                name="units_region",
                neurodata_type_inc="DynamicTableRegion",
                doc=(
                    "Reference to the units table rows that correspond to the units in this "
                    "sorting analysis. Order matches the unit dimension in all data arrays."
                ),
            ),
        ],
        links=[
            NWBLinkSpec(
                name="source_electrical_series",
                target_type="ElectricalSeries",
                quantity="?",
                doc=(
                    "Link to the ElectricalSeries from which this sorting was derived. "
                    "Optional - only present for 'with recording' cases."
                ),
            ),
        ],
        groups=[
            NWBGroupSpec(
                neurodata_type_inc="SpikeSortingExtensions",
                quantity="?",
                doc="Container for computed extension data (templates, random spikes, etc.).",
            ),
        ],
    )

    # Add all new data types
    new_data_types = [
        random_spikes,
        waveforms,
        templates,
        noise_levels,
        unit_locations,
        correlograms,
        isi_histograms,
        template_similarity,
        spike_amplitudes,
        spike_locations,
        amplitude_scalings,
        principal_components,
        spike_sorting_extensions,
        spike_sorting_container,
    ]

    # export the spec to yaml files in the root spec folder
    output_dir = str((Path(__file__).parent.parent.parent / "spec").absolute())
    export_spec(ns_builder, new_data_types, output_dir)


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()

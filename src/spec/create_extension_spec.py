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

    # RandomSpikesData: stores random spike indices per unit for waveform extraction
    random_spikes_data = NWBGroupSpec(
        neurodata_type_def="RandomSpikesData",
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

    # TemplatesData: stores sparse template waveforms per unit
    templates_data = NWBGroupSpec(
        neurodata_type_def="TemplatesData",
        neurodata_type_inc="NWBDataInterface",
        default_name="templates",
        doc=(
            "Sparse template waveforms per unit. Templates are stored as a ragged array where "
            "each unit has waveforms only for its active channels (determined by sparsity)."
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
                    "samples_after = n_samples - peak_sample_index - 1."
                ),
            ),
        ],
        datasets=[
            NWBDatasetSpec(
                name="data",
                neurodata_type_inc="VectorData",
                dtype="float32",
                dims=["total_channels", "num_samples"],
                shape=[None, None],
                doc=(
                    "Template waveforms stored sparsely. Shape is (total_active_channels_across_units, num_samples). "
                    "Use data_index to retrieve templates for each unit, and channel_ids to identify which "
                    "channel each waveform corresponds to."
                ),
                attributes=[
                    NWBAttributeSpec(
                        name="unit",
                        dtype="text",
                        default_value="volts",
                        doc="Unit of measurement for template data.",
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
                name="channel_ids",
                dtype="int32",
                dims=["total_channels"],
                shape=[None],
                doc=(
                    "Channel identifier for each row in data. Maps each waveform row to its "
                    "corresponding channel/electrode index."
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
                neurodata_type_inc="RandomSpikesData",
                quantity="?",
                doc="Random spikes extension data.",
            ),
            NWBGroupSpec(
                neurodata_type_inc="TemplatesData",
                quantity="?",
                doc="Templates extension data.",
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
    new_data_types = [random_spikes_data, templates_data, spike_sorting_extensions, spike_sorting_container]

    # export the spec to yaml files in the root spec folder
    output_dir = str((Path(__file__).parent.parent.parent / "spec").absolute())
    export_spec(ns_builder, new_data_types, output_dir)


if __name__ == "__main__":
    # usage: python create_extension_spec.py
    main()

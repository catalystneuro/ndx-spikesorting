from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hdmf.common import VectorData, VectorIndex, DynamicTableRegion
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries

if TYPE_CHECKING:
    import pandas as pd
    from spikeinterface.core import SortingAnalyzer

from ndx_spikesorting import (
    Templates,
    RandomSpikes,
    Waveforms,
    NoiseLevels,
    UnitLocations,
    Correlograms,
    ISIHistograms,
    TemplateSimilarity,
    SpikeAmplitudes,
    SpikeLocations,
    AmplitudeScalings,
    PCAProjectionsByChannel,
    PCAProjectionsConcatenated,
    FiringRate,
    UnitMetrics,
    ValidUnitPeriods,
    SpikeSortingExtensions,
    SpikeSortingContainer,
)

# Registries mapping canonical typed VectorData classes to
# (si_metric_name, si_extension_name). The writer uses col_cls=<class> when
# adding a column whose class appears here; other columns are written as plain
# VectorData with the name preserved. The registries are the single point of
# SpikeInterface-specific knowledge in the loader; the spec stays tool-agnostic.

UNITS_TYPED_COLUMNS = {
    FiringRate: ("firing_rate", "spiketrain_metrics"),
}

UNIT_METRICS_TYPED_COLUMNS: dict = {}


def templates_to_dense(templates: Templates, num_channels: int) -> np.ndarray:
    """Convert sparse ragged templates to a dense 3D array.

    Reconstructs a dense array of shape ``(num_units, num_samples, num_channels)``
    from the sparse ragged representation stored in NWB. Inactive channels are
    filled with zeros.

    Parameters
    ----------
    templates : Templates
        The Templates extension object containing the sparse template data.
    num_channels : int
        Total number of channels in the recording. Required because the
        sparse representation does not store inactive channels, so the
        total count cannot be inferred from the data alone.

    Returns
    -------
    numpy.ndarray
        Dense templates with shape ``(num_units, num_samples, num_channels)``,
        dtype float32.
    """
    sparse_data = templates.data.data[:]
    data_index = templates.data_index.data[:]
    electrode_indices = templates.electrodes.data[:]

    num_units = len(data_index)
    num_samples = sparse_data.shape[1]

    dense = np.zeros((num_units, num_samples, num_channels), dtype=np.float32)
    for unit_index in range(num_units):
        start = 0 if unit_index == 0 else data_index[unit_index - 1]
        end = data_index[unit_index]
        unit_sparse = sparse_data[start:end, :]
        active_channels = electrode_indices[start:end]

        for i, ch in enumerate(active_channels):
            dense[unit_index, :, ch] = unit_sparse[i, :]

    return dense


# ---------------------------------------------------------------------------
# Private helpers – one per SpikeInterface extension
# ---------------------------------------------------------------------------


def _convert_random_spikes(sorting_analyzer: "SortingAnalyzer") -> RandomSpikes:
    ext = sorting_analyzer.get_extension("random_spikes")
    random_spikes_data = ext.get_random_spikes()
    unit_ids = sorting_analyzer.unit_ids
    sorting = sorting_analyzer.sorting

    all_indices = []
    cumulative_index = []
    running_count = 0

    for unit_id in unit_ids:
        spike_train = sorting.get_unit_spike_train(unit_id=unit_id)
        unit_mask = random_spikes_data["unit_index"] == list(unit_ids).index(unit_id)
        unit_spike_indices = random_spikes_data["sample_index"][unit_mask]

        unit_indices = []
        for sample_index in unit_spike_indices:
            matches = np.where(spike_train == sample_index)[0]
            if len(matches) > 0:
                unit_indices.append(matches[0])

        unit_indices = np.array(unit_indices, dtype=np.int64)
        all_indices.append(unit_indices)
        running_count += len(unit_indices)
        cumulative_index.append(running_count)

    random_spikes_indices = VectorData(
        name="random_spikes_indices",
        data=np.concatenate(all_indices).astype(np.int64),
        description="Random spike indices for all units",
    )
    random_spikes_indices_index = VectorIndex(
        name="random_spikes_indices_index",
        data=np.array(cumulative_index, dtype=np.int64),
        target=random_spikes_indices,
    )
    return RandomSpikes(
        name="random_spikes",
        random_spikes_indices=random_spikes_indices,
        random_spikes_indices_index=random_spikes_indices_index,
    )


def _convert_waveforms(
    sorting_analyzer: "SortingAnalyzer",
    nwbfile: NWBFile,
    nwb_random_spikes: RandomSpikes,
    nbefore: int,
) -> Waveforms:
    ext = sorting_analyzer.get_extension("waveforms")
    unit_ids = sorting_analyzer.unit_ids
    sparsity = sorting_analyzer.sparsity

    all_wf_rows = []
    all_wf_electrode_indices = []
    wf_spike_cumulative = []
    wf_unit_cumulative = []

    total_spikes = 0
    running_row_count = 0
    for unit_id in unit_ids:
        unit_waveforms = ext.get_waveforms_one_unit(unit_id=unit_id, force_dense=False)
        n_spikes_unit = unit_waveforms.shape[0]

        if sparsity is not None:
            channel_indices = sparsity.unit_id_to_channel_indices[unit_id]
        else:
            channel_indices = np.arange(unit_waveforms.shape[2])

        for spike_idx in range(n_spikes_unit):
            spike_wf = unit_waveforms[spike_idx].T
            all_wf_rows.append(spike_wf)
            all_wf_electrode_indices.extend(channel_indices)
            running_row_count += len(channel_indices)
            wf_spike_cumulative.append(running_row_count)

        total_spikes += n_spikes_unit
        wf_unit_cumulative.append(total_spikes)

    wf_data = VectorData(
        name="data",
        data=np.vstack(all_wf_rows).astype(np.float32),
        description="Waveform data (one row per channel per spike)",
    )
    wf_data_index = VectorIndex(
        name="data_index",
        data=np.array(wf_spike_cumulative, dtype=np.int64),
        target=wf_data,
    )
    wf_data_index_index = VectorIndex(
        name="data_index_index",
        data=np.array(wf_unit_cumulative, dtype=np.int64),
        target=wf_data_index,
    )
    wf_electrodes = DynamicTableRegion(
        name="electrodes",
        data=list(int(i) for i in all_wf_electrode_indices),
        description="Electrode for each waveform row.",
        table=nwbfile.electrodes,
    )
    return Waveforms(
        name="waveforms",
        peak_sample_index=int(nbefore),
        data=wf_data,
        data_index=wf_data_index,
        data_index_index=wf_data_index_index,
        electrodes=wf_electrodes,
        random_spikes=nwb_random_spikes,
    )


def _convert_templates(
    sorting_analyzer: "SortingAnalyzer",
    nwbfile: NWBFile,
) -> Templates:
    ext = sorting_analyzer.get_extension("templates")
    unit_ids = sorting_analyzer.unit_ids
    sparsity = sorting_analyzer.sparsity
    nbefore = ext.nbefore

    all_data = []
    all_electrode_indices = []
    cumulative_index = []
    running_row_count = 0

    for unit_id in unit_ids:
        template = ext.get_unit_template(unit_id=unit_id)

        if sparsity is not None:
            channel_indices = sparsity.unit_id_to_channel_indices[unit_id]
            sparse_template = template[:, channel_indices].T
        else:
            channel_indices = np.arange(template.shape[1])
            sparse_template = template.T

        all_data.append(sparse_template)
        all_electrode_indices.extend(channel_indices)
        running_row_count += len(sparse_template)
        cumulative_index.append(running_row_count)

    data = VectorData(
        name="data",
        data=np.vstack(all_data).astype(np.float32),
        description="Template waveforms",
    )
    data_index = VectorIndex(
        name="data_index",
        data=np.array(cumulative_index, dtype=np.int64),
        target=data,
    )
    template_electrodes = DynamicTableRegion(
        name="electrodes",
        data=list(int(i) for i in all_electrode_indices),
        description="Electrode for each waveform row.",
        table=nwbfile.electrodes,
    )
    return Templates(
        name="templates",
        peak_sample_index=int(nbefore),
        data=data,
        data_index=data_index,
        electrodes=template_electrodes,
    )


def _convert_noise_levels(sorting_analyzer: "SortingAnalyzer") -> NoiseLevels:
    ext = sorting_analyzer.get_extension("noise_levels")
    return NoiseLevels(name="noise_levels", data=ext.get_data())


def _convert_unit_locations(sorting_analyzer: "SortingAnalyzer") -> UnitLocations:
    ext = sorting_analyzer.get_extension("unit_locations")
    return UnitLocations(name="unit_locations", data=ext.get_data())


def _convert_correlograms(sorting_analyzer: "SortingAnalyzer") -> Correlograms:
    ext = sorting_analyzer.get_extension("correlograms")
    ccgs, bin_edges = ext.get_data()
    return Correlograms(name="correlograms", data=ccgs, bin_edges=bin_edges)


def _convert_isi_histograms(sorting_analyzer: "SortingAnalyzer") -> ISIHistograms:
    ext = sorting_analyzer.get_extension("isi_histograms")
    isis, bin_edges = ext.get_data()
    return ISIHistograms(name="isi_histograms", data=isis, bin_edges=bin_edges)


def _convert_template_similarity(sorting_analyzer: "SortingAnalyzer") -> TemplateSimilarity:
    ext = sorting_analyzer.get_extension("template_similarity")
    return TemplateSimilarity(name="template_similarity", data=ext.get_data())


def _convert_spike_vector_extension(
    sorting_analyzer: "SortingAnalyzer",
    extension_name: str,
    nwb_class: type,
) -> SpikeAmplitudes | SpikeLocations | AmplitudeScalings:
    ext = sorting_analyzer.get_extension(extension_name)
    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    unit_indices = spike_vector["unit_index"]
    sort_order = np.argsort(unit_indices)
    cumulative_index = np.cumsum(np.bincount(unit_indices))

    all_data = ext.get_data()[sort_order]
    if all_data.dtype.names is not None:
        all_data = np.stack([all_data[name] for name in all_data.dtype.names], axis=1)

    data = VectorData(
        name="data",
        data=all_data,
        description=f"{extension_name} data",
    )
    data_index = VectorIndex(
        name="data_index",
        data=np.array(cumulative_index, dtype=np.int64),
        target=data,
    )
    return nwb_class(name=extension_name, data=data, data_index=data_index)


def _add_cell_intrinsic_columns_to_units(
    sorting_analyzer: "SortingAnalyzer",
    nwbfile: NWBFile,
) -> None:
    """Add canonical cell-intrinsic metric columns to ``nwbfile.units``.

    Pulls per-unit values from SI's ``template_metrics`` and ``spiketrain_metrics``
    DataFrames and writes them as typed VectorData columns on the existing
    ``nwbfile.units`` table. Each column carries its type tag (e.g. ``FiringRate``)
    so a reader can recover the canonical mapping back to SI extensions.

    The rows of ``nwbfile.units`` are assumed to be aligned with
    ``sorting_analyzer.unit_ids`` (neuroconv's ``add_sorting_to_nwbfile``
    preserves this order).
    """
    if nwbfile.units is None:
        return

    n_units = len(sorting_analyzer.unit_ids)
    if len(nwbfile.units) != n_units:
        raise ValueError(
            f"Mismatch between nwbfile.units ({len(nwbfile.units)} rows) and "
            f"sorting_analyzer.unit_ids ({n_units} units). Add the sorting to the "
            "NWB file before calling add_sorting_analyzer_to_nwbfile."
        )

    # Cache SI extension DataFrames by name so we only pull each once.
    si_extension_data: dict[str, "pd.DataFrame"] = {}
    for col_cls, (si_name, si_ext_name) in UNITS_TYPED_COLUMNS.items():
        if si_ext_name not in si_extension_data:
            ext = sorting_analyzer.get_extension(si_ext_name)
            si_extension_data[si_ext_name] = ext.get_data() if ext is not None else None
        df = si_extension_data[si_ext_name]
        if df is None or si_name not in df.columns:
            continue
        if si_name in nwbfile.units.colnames:
            continue  # already populated
        nwbfile.units.add_column(
            name=si_name,
            description=col_cls.__doc__ or si_name,
            data=df[si_name].to_numpy(),
            col_cls=col_cls,
        )


def _convert_unit_metrics(
    sorting_analyzer: "SortingAnalyzer",
    nwbfile: NWBFile,
) -> UnitMetrics | None:
    """Build a UnitMetrics instance from SI's ``quality_metrics`` extension.

    Each row of the table references one unit (via the ``unit``
    ``DynamicTableRegion``) and carries that unit's run-dependent metric values
    as typed columns. If ``valid_unit_periods`` is also computed, the per-unit
    valid windows are written as ``computation_intervals`` on the table.
    Returns ``None`` when ``quality_metrics`` is not computed.
    """
    qm_ext = sorting_analyzer.get_extension("quality_metrics")
    if qm_ext is None:
        return None

    df = qm_ext.get_data()
    n_units = len(sorting_analyzer.unit_ids)
    qm_params = qm_ext.params or {}

    # Per-row unit DTR pointing back to nwbfile.units (one row per unit).
    unit_column = DynamicTableRegion(
        name="unit",
        data=list(range(n_units)),
        description="Reference to nwbfile.units row this metric value applies to.",
        table=nwbfile.units,
    )

    columns = [unit_column]

    # Track which DataFrame columns we have already written via typed-column
    # paths so we do not duplicate them as plain VectorData below. Currently
    # empty (no run-dependent typed columns committed in v1), but keeps the
    # design open for adding typed columns in a later version.
    typed_column_names: set[str] = set()
    for col_cls, (si_name, _) in UNIT_METRICS_TYPED_COLUMNS.items():
        if si_name not in df.columns:
            continue
        col_kwargs = _build_column_kwargs(col_cls, si_name, df[si_name].to_numpy(), qm_params)
        columns.append(col_cls(**col_kwargs))
        typed_column_names.add(si_name)

    # The remaining quality_metrics DataFrame columns become plain VectorData
    # on the UnitMetrics instance. Column names are preserved so the loader can
    # reconstruct SI's quality_metrics extension on read.
    for col_name in df.columns:
        if col_name in typed_column_names:
            continue
        col_values = df[col_name].to_numpy()
        # Coerce to float for storage; NaNs are preserved.
        columns.append(
            VectorData(
                name=col_name,
                data=col_values.astype(float),
                description=f"Run-dependent metric: {col_name}.",
            )
        )

    vup_ext = sorting_analyzer.get_extension("valid_unit_periods")
    if vup_ext is not None:
        if sorting_analyzer.get_num_segments() > 1:
            raise NotImplementedError(
                "valid_unit_periods round-trip is single-segment only; "
                "multi-segment support not yet implemented."
            )
        sampling_frequency = sorting_analyzer.sampling_frequency
        valid_periods_data = vup_ext.get_data(outputs="numpy")
        per_unit_windows: list[list[tuple[float, float]]] = [[] for _ in range(n_units)]
        for period in valid_periods_data:
            unit_index = int(period["unit_index"])
            start_s = float(period["start_sample_index"]) / sampling_frequency
            stop_s = float(period["end_sample_index"]) / sampling_frequency
            per_unit_windows[unit_index].append((start_s, stop_s))

        flat_intervals: list[list[float]] = []
        cumulative = []
        running = 0
        for windows in per_unit_windows:
            for start_s, stop_s in windows:
                flat_intervals.append([start_s, stop_s])
            running += len(windows)
            cumulative.append(running)

        computation_intervals_vd = VectorData(
            name="computation_intervals",
            data=np.array(flat_intervals, dtype=np.float64) if flat_intervals else np.zeros((0, 2)),
            description=(
                "Per-unit time intervals (start, stop) in seconds over which this row's "
                "metric values were computed."
            ),
        )
        computation_intervals_index = VectorIndex(
            name="computation_intervals_index",
            data=np.array(cumulative, dtype=np.int64),
            target=computation_intervals_vd,
        )
        columns.append(computation_intervals_vd)
        columns.append(computation_intervals_index)

    return UnitMetrics(
        name="quality_metrics",
        description="Run-dependent per-unit metrics from one analysis run.",
        columns=columns,
    )


def _convert_valid_unit_periods(
    sorting_analyzer: "SortingAnalyzer",
    units_table=None,
) -> ValidUnitPeriods | None:
    ext = sorting_analyzer.get_extension("valid_unit_periods")
    if ext is None:
        return None
    valid_periods_data = ext.get_data(outputs="numpy")
    sampling_frequency = sorting_analyzer.sampling_frequency

    start_times = []
    stop_times = []
    unit_indices = []

    for period in valid_periods_data:
        start_times.append(float(period["start_sample_index"]) / sampling_frequency)
        stop_times.append(float(period["end_sample_index"]) / sampling_frequency)
        unit_indices.append(int(period["unit_index"]))

    vup_unit_column = DynamicTableRegion(
        name="unit",
        data=unit_indices,
        description="Reference to units table for each valid period.",
        table=units_table,
    )

    return ValidUnitPeriods(
        name="valid_unit_periods",
        description="Valid time periods per unit from spike sorting quality estimation.",
        columns=[
            VectorData(
                name="start_time", data=start_times, description="Start time of each valid period in seconds."
            ),
            VectorData(
                name="stop_time", data=stop_times, description="Stop time of each valid period in seconds."
            ),
            vup_unit_column,
        ],
    )


def _build_column_kwargs(
    col_cls: type,
    si_name: str,
    data: np.ndarray,
    qm_params: dict,
) -> dict:
    """Assemble constructor kwargs for a typed canonical run-dependent column.

    Currently a stub: v1 does not commit any run-dependent typed columns, so
    this is unused. It is kept as the place where attribute-extraction logic
    will live when typed run-dependent columns are added in a future version
    (e.g. pulling SI's ``isi_threshold_ms`` onto a future
    ``IsiViolationsRatio.refractory_period_ms`` attribute).
    """
    return dict(
        name=si_name,
        description=col_cls.__doc__ or si_name,
        data=data,
    )


def _convert_pca_by_channel(
    sorting_analyzer: "SortingAnalyzer",
    nwbfile: NWBFile,
    nwb_waveforms: Waveforms,
) -> PCAProjectionsByChannel:
    ext = sorting_analyzer.get_extension("principal_components")
    unit_ids = sorting_analyzer.unit_ids
    sparsity = sorting_analyzer.sparsity

    all_pc_rows = []
    all_pc_electrode_indices = []
    pc_spike_cumulative = []
    pc_unit_cumulative = []

    total_spikes = 0
    running_row_count = 0
    for unit_id in unit_ids:
        unit_pcs = ext.get_projections_one_unit(unit_id=unit_id, sparse=True)
        if isinstance(unit_pcs, tuple):
            unit_pcs, _ = unit_pcs
        n_spikes_unit = unit_pcs.shape[0]

        if sparsity is not None:
            channel_indices = sparsity.unit_id_to_channel_indices[unit_id]
        else:
            channel_indices = np.arange(unit_pcs.shape[2])

        for spike_idx in range(n_spikes_unit):
            spike_pc = unit_pcs[spike_idx].T
            all_pc_rows.append(spike_pc)
            all_pc_electrode_indices.extend(channel_indices)
            running_row_count += len(channel_indices)
            pc_spike_cumulative.append(running_row_count)

        total_spikes += n_spikes_unit
        pc_unit_cumulative.append(total_spikes)

    pc_data = VectorData(
        name="data",
        data=np.vstack(all_pc_rows).astype(np.float64),
        description="PCA projection data",
    )
    pc_data_index = VectorIndex(
        name="data_index",
        data=np.array(pc_spike_cumulative, dtype=np.int64),
        target=pc_data,
    )
    pc_data_index_index = VectorIndex(
        name="data_index_index",
        data=np.array(pc_unit_cumulative, dtype=np.int64),
        target=pc_data_index,
    )
    pc_electrodes = DynamicTableRegion(
        name="electrodes",
        data=list(int(i) for i in all_pc_electrode_indices),
        description="Electrode for each projection row.",
        table=nwbfile.electrodes,
    )
    return PCAProjectionsByChannel(
        name="pca_projections_by_channel",
        data=pc_data,
        data_index=pc_data_index,
        data_index_index=pc_data_index_index,
        electrodes=pc_electrodes,
        waveforms=nwb_waveforms,
    )


def _convert_pca_concatenated(
    sorting_analyzer: "SortingAnalyzer",
    nwb_waveforms: Waveforms,
) -> PCAProjectionsConcatenated:
    ext = sorting_analyzer.get_extension("principal_components")
    unit_ids = sorting_analyzer.unit_ids

    all_pc_rows = []
    pc_unit_cumulative = []

    total_spikes = 0
    for unit_id in unit_ids:
        unit_pcs = ext.get_projections_one_unit(unit_id=unit_id, sparse=True)
        if isinstance(unit_pcs, tuple):
            unit_pcs, _ = unit_pcs
        n_spikes_unit = unit_pcs.shape[0]
        all_pc_rows.append(unit_pcs)
        total_spikes += n_spikes_unit
        pc_unit_cumulative.append(total_spikes)

    pc_data = VectorData(
        name="data",
        data=np.vstack(all_pc_rows).astype(np.float64),
        description="PCA projection data",
    )
    pc_data_index = VectorIndex(
        name="data_index",
        data=np.array(pc_unit_cumulative, dtype=np.int64),
        target=pc_data,
    )
    return PCAProjectionsConcatenated(
        name="pca_projections_concatenated",
        data=pc_data,
        data_index=pc_data_index,
        waveforms=nwb_waveforms,
    )


# ---------------------------------------------------------------------------
# Extension assembly helpers
# ---------------------------------------------------------------------------


def _convert_pca(sorting_analyzer, nwbfile, nwb_waveforms):
    """Convert PCA extension, returning a dict with the correct attribute key."""
    pc_ext = sorting_analyzer.get_extension("principal_components")
    pc_mode = pc_ext.params.get("mode", "by_channel_local")
    if pc_mode in ("by_channel_local", "by_channel_global"):
        return {"pca_projections_by_channel": _convert_pca_by_channel(sorting_analyzer, nwbfile, nwb_waveforms)}
    return {"pca_projections_concatenated": _convert_pca_concatenated(sorting_analyzer, nwb_waveforms)}


def _convert_all_extensions(sorting_analyzer, nwbfile):  # noqa: C901
    """Convert computed SortingAnalyzer extensions to NWB objects.

    Returns a dict mapping attribute names on ``SpikeSortingExtensions``
    to the converted NWB objects.
    """

    def _has(name: str) -> bool:
        return sorting_analyzer.get_extension(name) is not None

    converted = {}

    if _has("random_spikes"):
        converted["random_spikes"] = _convert_random_spikes(sorting_analyzer)

    templates_ext = sorting_analyzer.get_extension("templates")
    if templates_ext is not None:
        converted["templates"] = _convert_templates(sorting_analyzer, nwbfile)
    nbefore = templates_ext.nbefore if templates_ext is not None else None

    if _has("waveforms"):
        if "random_spikes" not in converted:
            raise ValueError("Waveforms extension requires random_spikes to be computed.")
        if nbefore is None:
            raise ValueError("Waveforms extension requires templates to be computed (for nbefore).")
        converted["waveforms"] = _convert_waveforms(
            sorting_analyzer, nwbfile, converted["random_spikes"], nbefore
        )

    if _has("noise_levels"):
        converted["noise_levels"] = _convert_noise_levels(sorting_analyzer)
    if _has("unit_locations"):
        converted["unit_locations"] = _convert_unit_locations(sorting_analyzer)
    if _has("correlograms"):
        converted["correlograms"] = _convert_correlograms(sorting_analyzer)
    if _has("isi_histograms"):
        converted["isi_histograms"] = _convert_isi_histograms(sorting_analyzer)
    if _has("template_similarity"):
        converted["template_similarity"] = _convert_template_similarity(sorting_analyzer)
    if _has("spike_amplitudes"):
        converted["spike_amplitudes"] = _convert_spike_vector_extension(
            sorting_analyzer, "spike_amplitudes", SpikeAmplitudes
        )
    if _has("spike_locations"):
        converted["spike_locations"] = _convert_spike_vector_extension(
            sorting_analyzer, "spike_locations", SpikeLocations
        )
    if _has("amplitude_scalings"):
        converted["amplitude_scalings"] = _convert_spike_vector_extension(
            sorting_analyzer, "amplitude_scalings", AmplitudeScalings
        )
    if _has("principal_components"):
        if "waveforms" not in converted:
            raise ValueError("PCA projections require waveforms to be computed.")
        converted.update(_convert_pca(sorting_analyzer, nwbfile, converted["waveforms"]))

    # Run-dependent metrics flow into a UnitMetrics instance (or several, as
    # support for multiple curation runs is added). For v1 there is one
    # UnitMetrics per file, sourced from SI's quality_metrics extension.
    unit_metrics = _convert_unit_metrics(sorting_analyzer, nwbfile)
    if unit_metrics is not None:
        converted["__unit_metrics__"] = unit_metrics

    vup = _convert_valid_unit_periods(sorting_analyzer, units_table=nwbfile.units)
    if vup is not None:
        converted["valid_unit_periods"] = vup

    return converted


def _build_extensions(sorting_analyzer, nwbfile):
    """Convert extensions and assemble a SpikeSortingExtensions object.

    Cell-intrinsic metric columns are added directly to ``nwbfile.units``
    (a side effect) rather than to the extensions container, because they
    are per-unit properties of the cells themselves.
    """
    # Side effect: write cell-intrinsic metric columns onto nwbfile.units.
    _add_cell_intrinsic_columns_to_units(sorting_analyzer, nwbfile)

    converted = _convert_all_extensions(sorting_analyzer, nwbfile)
    extensions = SpikeSortingExtensions(name="extensions")
    for attr, obj in converted.items():
        if attr == "__unit_metrics__":
            extensions.add_unit_metrics(obj)
        else:
            setattr(extensions, attr, obj)
    return extensions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_sorting_analyzer_to_nwbfile(
    sorting_analyzer: "SortingAnalyzer",
    nwbfile: NWBFile,
    electrical_series_name: str | None = None,
    unit_table_name: str | None = None,
):
    """Add sorting analyzer extensions to an NWBFile.

    Converts computed SpikeInterface extensions from a ``SortingAnalyzer``
    into ndx-spikesorting NWB types and adds them inside a
    ``SpikeSortingContainer`` under the ``ecephys`` processing module.

    The *nwbfile* must already contain an electrodes table and a units table
    (e.g. added via ``neuroconv``).

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SpikeInterface ``SortingAnalyzer`` with one or more computed
        extensions.
    nwbfile : NWBFile
        The target NWB file. Must already have electrodes and units.
    electrical_series_name : str or None, optional
        Name of the ``ElectricalSeries`` in ``nwbfile.acquisition`` to link
        as the source recording. If *None* and exactly one
        ``ElectricalSeries`` exists it is used automatically. If *None* and
        more than one exist, a ``ValueError`` is raised. If no
        ``ElectricalSeries`` exists the link is omitted.
    unit_table_name : str or None, optional
        Name of the units ``DynamicTable``. If *None* the default
        ``nwbfile.units`` is used. Raises ``ValueError`` when
        ``nwbfile.units`` is *None*.
    """
    if sorting_analyzer.get_num_segments() > 1:
        raise NotImplementedError("Currently only single-segment SortingAnalyzers are supported.")

    # -- resolve electrical series --
    electrical_series = None
    es_candidates = {
        name: obj
        for name, obj in (nwbfile.acquisition or {}).items()
        if isinstance(obj, ElectricalSeries)
    }
    if electrical_series_name is not None:
        electrical_series = nwbfile.acquisition[electrical_series_name]
    elif len(es_candidates) == 1:
        electrical_series = next(iter(es_candidates.values()))
    elif len(es_candidates) > 1:
        raise ValueError(
            f"Multiple ElectricalSeries found in nwbfile.acquisition "
            f"({list(es_candidates)}). Specify electrical_series_name explicitly."
        )

    # -- resolve units table --
    if unit_table_name is not None:
        units_table = nwbfile.processing.get(unit_table_name) or nwbfile.units
    else:
        units_table = nwbfile.units
    if units_table is None:
        raise ValueError("nwbfile has no units table. Add units before calling this function.")

    # -- build electrode & unit regions --
    num_channels = sorting_analyzer.get_num_channels()
    electrodes_region = nwbfile.create_electrode_table_region(
        region=list(range(num_channels)),
        description="All electrodes used in sorting analysis",
    )

    num_units = len(sorting_analyzer.unit_ids)
    units_region = DynamicTableRegion(
        name="units_region",
        data=list(range(num_units)),
        description="All units from sorting analysis",
        table=units_table,
    )

    # -- convert & assemble extensions --
    extensions = _build_extensions(sorting_analyzer, nwbfile)

    # -- assemble SpikeSortingContainer --
    sparsity = sorting_analyzer.sparsity
    sparsity_mask = sparsity.mask if sparsity is not None else None

    container_kwargs = dict(
        name="spike_sorting",
        sampling_frequency=sorting_analyzer.sampling_frequency,
        electrodes=electrodes_region,
        units_region=units_region,
        sparsity_mask=sparsity_mask,
    )
    if electrical_series is not None:
        container_kwargs["source_electrical_series"] = electrical_series

    container = SpikeSortingContainer(**container_kwargs)
    container.spike_sorting_extensions = extensions

    # -- add to processing module --
    if "ecephys" in nwbfile.processing:
        ecephys_module = nwbfile.processing["ecephys"]
    else:
        ecephys_module = nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing results",
        )
    ecephys_module.add(container)


def read_sorting_analyzer_from_nwb(
    nwbfile_path: str | Path,
    container_path: str = "ecephys/spike_sorting",
) -> "SortingAnalyzer":
    """Read an ndx-spikesorting NWB file and return a SortingAnalyzer.

    Opens the NWB file, extracts the ``SpikeSortingContainer`` at
    *container_path* inside ``processing``, reconstructs sorting/recording
    objects, and instantiates any precomputed extensions.

    Parameters
    ----------
    nwbfile_path : str or Path
        Path to the NWB file on disk.
    container_path : str, optional
        Slash-separated path to the ``SpikeSortingContainer`` inside
        ``nwbfile.processing``, e.g. ``"ecephys/spike_sorting"``.
        Defaults to ``"ecephys/spike_sorting"``.

    Returns
    -------
    SortingAnalyzer
        In-memory SortingAnalyzer with precomputed extensions instantiated.
    """
    from pathlib import Path

    from pynwb import NWBHDF5IO
    from spikeinterface.core import ChannelSparsity, create_sorting_analyzer
    from spikeinterface.extractors import NwbRecordingExtractor, NwbSortingExtractor

    nwbfile_path = Path(nwbfile_path)

    # -- Open NWB and locate the container --
    with NWBHDF5IO(nwbfile_path, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()

        parts = container_path.split("/")
        obj = nwbfile.processing[parts[0]]
        for part in parts[1:]:
            obj = obj[part]
        container = obj
        sampling_frequency = container.sampling_frequency

        # -- Load sorting --
        sorting = NwbSortingExtractor(
            file_path=nwbfile_path,
            sampling_frequency=sampling_frequency,
            t_start=0.0,
        )

        # -- Load recording (if linked) --
        recording = None
        if container.source_electrical_series is not None:
            es = container.source_electrical_series
            es_path = f"acquisition/{es.name}"
            recording = NwbRecordingExtractor(file_path=nwbfile_path, electrical_series_path=es_path)

        # -- Build sparsity from stored mask --
        sparsity = None
        if container.sparsity_mask is not None:
            channel_ids = recording.channel_ids if recording is not None else np.array(container.electrodes.data[:])
            sparsity = ChannelSparsity(
                mask=np.array(container.sparsity_mask[:], dtype=bool),
                unit_ids=sorting.unit_ids,
                channel_ids=channel_ids,
            )

        # -- Create in-memory SortingAnalyzer --
        sorting_analyzer = create_sorting_analyzer(
            sorting=sorting,
            recording=recording,
            format="memory",
            sparse=False,
            sparsity=sparsity,
        )

        # -- Instantiate precomputed extensions --
        extensions = container.spike_sorting_extensions
        if extensions is not None:
            _load_random_spikes(extensions, sorting_analyzer)
            _load_waveforms(extensions, sorting_analyzer)
            _load_templates(extensions, sorting_analyzer)
            _load_noise_levels(extensions, sorting_analyzer)
            _load_unit_locations(extensions, sorting_analyzer)
            _load_correlograms(extensions, sorting_analyzer)
            _load_isi_histograms(extensions, sorting_analyzer)
            _load_template_similarity(extensions, sorting_analyzer)
            _load_spike_amplitudes(extensions, sorting_analyzer)
            _load_amplitude_scalings(extensions, sorting_analyzer)
            _load_spike_locations(extensions, sorting_analyzer)
            _load_pca_projections(extensions, sorting_analyzer)
            _load_unit_metrics(extensions, sorting_analyzer)
            _load_valid_unit_periods_from_nwb(extensions, sorting_analyzer)

        # Cell-intrinsic metrics live on nwbfile.units as typed columns; read
        # them by column type so the SI extension binding survives even if a
        # column was renamed.
        if nwbfile.units is not None:
            _load_cell_intrinsic_from_units(nwbfile.units, sorting_analyzer)

    return sorting_analyzer


# ---------------------------------------------------------------------------
# Private helpers – NWB → SortingAnalyzer extension loading
# ---------------------------------------------------------------------------


def _load_random_spikes(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    """Instantiate the random_spikes extension if present in the NWB container.

    Converts per-unit local spike train indices (NWB ragged representation)
    back to global spike vector indices (SpikeInterface representation).
    """
    from spikeinterface.core.sorting_tools import spike_vector_to_indices
    from spikeinterface.core.sortinganalyzer import get_extension_class

    random_spikes_nwb = extensions.random_spikes
    if random_spikes_nwb is None:
        return

    sorting = sorting_analyzer.sorting
    indices_data = random_spikes_nwb.random_spikes_indices.data[:]
    index_boundaries = random_spikes_nwb.random_spikes_indices_index.data[:]

    unit_ids = sorting_analyzer.unit_ids
    per_unit_counts = np.diff(np.concatenate([[0], index_boundaries]))

    total_spike_counts = sorting.count_num_spikes_per_unit(outputs="array")
    method = "all" if np.array_equal(per_unit_counts, total_spike_counts) else "uniform"
    max_spikes_per_unit = int(per_unit_counts.max())

    spike_vector = sorting.to_spike_vector(concatenated=False)
    spike_indices_by_unit_and_segments = spike_vector_to_indices(spike_vector, unit_ids, absolute_index=True)
    segment_index = 0
    spike_indices_by_unit = spike_indices_by_unit_and_segments[segment_index]

    global_indices = []
    for unit_index, unit_id in enumerate(unit_ids):
        start = 0 if unit_index == 0 else index_boundaries[unit_index - 1]
        end = index_boundaries[unit_index]
        unit_spike_train_indices = indices_data[start:end]
        global_indices.extend(spike_indices_by_unit[unit_id][unit_spike_train_indices])

    random_spikes_indices = np.array(sorted(global_indices), dtype=np.int64)

    ext_class = get_extension_class("random_spikes")
    ext = ext_class(sorting_analyzer)
    ext.set_params(method=method, max_spikes_per_unit=max_spikes_per_unit)
    ext.data["random_spikes_indices"] = random_spikes_indices
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["random_spikes"] = ext


def _load_waveforms(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    """Instantiate the waveforms extension if present in the NWB container.

    Converts the double-ragged NWB representation (rows grouped by spike,
    spikes grouped by unit) back to SpikeInterface's flat array indexed by
    the random_spikes extension's spike ordering.
    """
    from spikeinterface.core.sortinganalyzer import get_extension_class

    waveforms_nwb = extensions.waveforms
    if waveforms_nwb is None:
        return

    sparse_data = waveforms_nwb.data.data[:]
    data_index = waveforms_nwb.data_index.data[:]
    data_index_index = waveforms_nwb.data_index_index.data[:]
    num_samples = sparse_data.shape[1]
    num_units = len(data_index_index)

    per_unit_waveforms = []
    max_sparse_channels = 0
    for unit_idx in range(num_units):
        spike_start = 0 if unit_idx == 0 else int(data_index_index[unit_idx - 1])
        spike_end = int(data_index_index[unit_idx])

        unit_waveforms = []
        for spike_idx in range(spike_start, spike_end):
            row_start = 0 if spike_idx == 0 else int(data_index[spike_idx - 1])
            row_end = int(data_index[spike_idx])
            n_channels = row_end - row_start
            max_sparse_channels = max(max_sparse_channels, n_channels)
            spike_wf = sparse_data[row_start:row_end, :].T
            unit_waveforms.append(spike_wf)

        if unit_waveforms:
            per_unit_waveforms.append(np.stack(unit_waveforms))
        else:
            per_unit_waveforms.append(np.empty((0, num_samples, 0), dtype=sparse_data.dtype))

    some_spikes = sorting_analyzer.get_extension("random_spikes").get_random_spikes()
    total_spikes = len(some_spikes)
    all_waveforms = np.zeros((total_spikes, num_samples, max_sparse_channels), dtype=np.float32)

    for unit_idx in range(num_units):
        spike_mask = some_spikes["unit_index"] == unit_idx
        unit_wfs = per_unit_waveforms[unit_idx]
        if unit_wfs.shape[0] > 0:
            n_ch = unit_wfs.shape[2]
            all_waveforms[spike_mask, :, :n_ch] = unit_wfs

    ext_class = get_extension_class("waveforms")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["waveforms"] = all_waveforms
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["waveforms"] = ext


def _load_templates(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    """Instantiate the templates extension if present in the NWB container.

    Converts the sparse ragged NWB representation back to SpikeInterface's
    dense ``(n_units, n_samples, n_channels)`` array.
    """
    from spikeinterface.core.sortinganalyzer import get_extension_class

    templates_nwb = extensions.templates
    if templates_nwb is None:
        return

    peak_sample_index = templates_nwb.peak_sample_index
    num_channels = sorting_analyzer.get_num_channels()
    num_samples = templates_nwb.data.data.shape[1]

    ms_before = peak_sample_index / sorting_analyzer.sampling_frequency * 1000.0
    ms_after = (num_samples - peak_sample_index) / sorting_analyzer.sampling_frequency * 1000.0

    dense_templates = templates_to_dense(templates_nwb, num_channels)

    ext_class = get_extension_class("templates")
    ext = ext_class(sorting_analyzer)
    ext.set_params(ms_before=float(ms_before), ms_after=float(ms_after), operators=["average", "std"])
    ext.data["average"] = dense_templates
    ext.data["std"] = np.zeros_like(dense_templates)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["templates"] = ext


def _load_noise_levels(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    from spikeinterface.core.sortinganalyzer import get_extension_class

    noise_levels_nwb = extensions.noise_levels
    if noise_levels_nwb is None:
        return

    ext_class = get_extension_class("noise_levels")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["noise_levels"] = noise_levels_nwb.data[:].astype(np.float32)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["noise_levels"] = ext


def _load_unit_locations(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    from spikeinterface.core.sortinganalyzer import get_extension_class

    unit_locations_nwb = extensions.unit_locations
    if unit_locations_nwb is None:
        return

    ext_class = get_extension_class("unit_locations")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["unit_locations"] = unit_locations_nwb.data[:]
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["unit_locations"] = ext


def _load_correlograms(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    from spikeinterface.core.sortinganalyzer import get_extension_class

    correlograms_nwb = extensions.correlograms
    if correlograms_nwb is None:
        return

    ext_class = get_extension_class("correlograms")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["ccgs"] = correlograms_nwb.data[:]
    ext.data["bins"] = correlograms_nwb.bin_edges[:]
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["correlograms"] = ext


def _load_isi_histograms(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    from spikeinterface.core.sortinganalyzer import get_extension_class

    isi_histograms_nwb = extensions.isi_histograms
    if isi_histograms_nwb is None:
        return

    ext_class = get_extension_class("isi_histograms")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["isi_histograms"] = isi_histograms_nwb.data[:]
    ext.data["bins"] = isi_histograms_nwb.bin_edges[:]
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["isi_histograms"] = ext


def _load_template_similarity(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    from spikeinterface.core.sortinganalyzer import get_extension_class

    template_similarity_nwb = extensions.template_similarity
    if template_similarity_nwb is None:
        return

    ext_class = get_extension_class("template_similarity")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["similarity"] = template_similarity_nwb.data[:]
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["template_similarity"] = ext


def _load_spike_amplitudes(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    from spikeinterface.core.sortinganalyzer import get_extension_class

    spike_amplitudes_nwb = extensions.spike_amplitudes
    if spike_amplitudes_nwb is None:
        return

    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    unit_indices = spike_vector["unit_index"]
    sort_order = np.argsort(unit_indices)
    reverse_order = np.argsort(sort_order, kind="stable")

    ext_class = get_extension_class("spike_amplitudes")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["amplitudes"] = spike_amplitudes_nwb.data[:][reverse_order].astype(np.float32)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["spike_amplitudes"] = ext


def _load_spike_locations(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    from spikeinterface.core.sortinganalyzer import get_extension_class

    spike_locations_nwb = extensions.spike_locations
    if spike_locations_nwb is None:
        return

    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    unit_indices = spike_vector["unit_index"]
    sort_order = np.argsort(unit_indices)
    reverse_order = np.argsort(sort_order, kind="stable")
    spike_locations = spike_locations_nwb.data[:][reverse_order]

    if spike_locations.shape[1] == 2:
        spike_locations = np.rec.fromarrays(spike_locations.T, names="x,y")
    elif spike_locations.shape[1] == 3:
        spike_locations = np.rec.fromarrays(spike_locations.T, names="x,y,z")

    ext_class = get_extension_class("spike_locations")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["spike_locations"] = spike_locations
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["spike_locations"] = ext


def _load_amplitude_scalings(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    from spikeinterface.core.sortinganalyzer import get_extension_class

    amplitude_scalings_nwb = extensions.amplitude_scalings
    if amplitude_scalings_nwb is None:
        return

    spike_vector = sorting_analyzer.sorting.to_spike_vector()
    unit_indices = spike_vector["unit_index"]
    sort_order = np.argsort(unit_indices)
    reverse_order = np.argsort(sort_order, kind="stable")

    ext_class = get_extension_class("amplitude_scalings")
    ext = ext_class(sorting_analyzer)
    ext.set_params()
    ext.data["amplitude_scalings"] = amplitude_scalings_nwb.data[:][reverse_order].astype(np.float32)
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["amplitude_scalings"] = ext


def _load_cell_intrinsic_from_units(units_table, sorting_analyzer: "SortingAnalyzer") -> None:
    """Read canonical cell-intrinsic typed columns from a Units table.

    Walks the Units table columns, finds those that are instances of canonical
    typed VectorData classes, and populates the corresponding SI extension
    DataFrames. Routes by column class via ``UNITS_TYPED_COLUMNS``.

    When a target SI extension was already populated by ``_load_unit_metrics``
    (e.g. quality_metrics receives both ``Snr`` from a UnitMetrics and
    ``AmplitudeMedian`` from a Units column), this function merges into the
    existing DataFrame rather than overwriting it.
    """
    import pandas as pd
    from spikeinterface.core.sortinganalyzer import get_extension_class

    per_extension_columns: dict[str, dict[str, np.ndarray]] = {}
    for col_name in units_table.colnames:
        col = units_table[col_name]
        for col_cls, (si_name, si_ext_name) in UNITS_TYPED_COLUMNS.items():
            if isinstance(col, col_cls):
                per_extension_columns.setdefault(si_ext_name, {})[si_name] = np.asarray(col.data[:])
                break

    for si_ext_name, columns_dict in per_extension_columns.items():
        existing = sorting_analyzer.extensions.get(si_ext_name)
        if existing is not None:
            # Merge into existing DataFrame; preserve any params/metric_names set
            # by the prior loader (typically _load_unit_metrics for quality_metrics).
            existing_df = existing.data.get("metrics")
            if existing_df is not None:
                for col_name, values in columns_dict.items():
                    existing_df[col_name] = values
                existing.params.setdefault("metric_names", []).extend(
                    [n for n in columns_dict if n not in existing.params["metric_names"]]
                )
                existing.params["metrics_to_compute"] = list(existing.params["metric_names"])
                continue

        ext_class = get_extension_class(si_ext_name)
        if ext_class is None:
            continue
        unit_ids = sorting_analyzer.unit_ids
        metrics_df = pd.DataFrame(columns_dict, index=unit_ids)
        ext = ext_class(sorting_analyzer)
        ext.params = {
            "metric_names": list(columns_dict.keys()),
            "metric_params": {},
            "metrics_to_compute": list(columns_dict.keys()),
            "delete_existing_metrics": False,
        }
        ext.data["metrics"] = metrics_df
        ext.run_info = {"run_completed": True, "runtime_s": 0.0}
        sorting_analyzer.extensions[si_ext_name] = ext


def _load_unit_metrics(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    """Read UnitMetrics instances and reconstruct SI's ``quality_metrics`` extension.

    Each UnitMetrics is a curation run whose typed (and plain) columns populate
    ``quality_metrics``. For v1 we expect at most one UnitMetrics per file and
    route it directly. Multi-run support (each run distinguished by name or
    parameter context) is a follow-up.
    """
    import pandas as pd
    from spikeinterface.core.sortinganalyzer import get_extension_class

    unit_metrics = getattr(extensions, "unit_metrics", None)
    if not unit_metrics:
        return

    # Iterate over UnitMetrics instances (LabelledDict keyed by name)
    runs = unit_metrics.values() if hasattr(unit_metrics, "values") else [unit_metrics]
    for nwb_table in runs:
        unit_ids = sorting_analyzer.unit_ids
        per_extension_columns: dict[str, dict[str, np.ndarray]] = {}
        per_extension_params: dict[str, dict] = {}

        for col_name in nwb_table.colnames:
            if col_name in ("unit", "computation_intervals", "computation_intervals_index"):
                continue
            col = nwb_table[col_name]
            # First try the typed-column registry (empty in v1; kept for forward
            # compatibility when run-dependent typed columns are added).
            matched = False
            for col_cls, (si_name, si_ext_name) in UNIT_METRICS_TYPED_COLUMNS.items():
                if isinstance(col, col_cls):
                    per_extension_columns.setdefault(si_ext_name, {})[si_name] = np.asarray(col.data[:])
                    matched = True
                    break
            if matched:
                continue
            # Fallback: plain VectorData column. Route to quality_metrics by
            # convention, using the column name as the SI metric name.
            per_extension_columns.setdefault("quality_metrics", {})[col_name] = np.asarray(col.data[:])

        for si_ext_name, columns_dict in per_extension_columns.items():
            ext_class = get_extension_class(si_ext_name)
            if ext_class is None:
                continue
            metrics_df = pd.DataFrame(columns_dict, index=unit_ids)
            params = per_extension_params.get(si_ext_name, {})
            ext = ext_class(sorting_analyzer)
            ext.params = {
                "metric_names": list(columns_dict.keys()),
                "metric_params": params.get("metric_params", {}),
                "metrics_to_compute": list(columns_dict.keys()),
                "delete_existing_metrics": False,
            }
            ext.data["metrics"] = metrics_df
            ext.run_info = {"run_completed": True, "runtime_s": 0.0}
            sorting_analyzer.extensions[si_ext_name] = ext


def _load_valid_unit_periods_from_nwb(extensions, sorting_analyzer):
    from spikeinterface.core.base import unit_period_dtype
    from spikeinterface.core.sortinganalyzer import get_extension_class

    valid_periods_nwb = getattr(extensions, "valid_unit_periods", None)
    if valid_periods_nwb is None:
        return

    start_times = np.array(valid_periods_nwb["start_time"][:], dtype=np.float64)
    stop_times = np.array(valid_periods_nwb["stop_time"][:], dtype=np.float64)
    unit_indices = np.array(valid_periods_nwb["unit"].data[:], dtype=np.int64)

    sampling_frequency = sorting_analyzer.sampling_frequency
    n_periods = len(start_times)

    valid_periods = np.zeros(n_periods, dtype=unit_period_dtype)
    valid_periods["segment_index"] = 0
    valid_periods["start_sample_index"] = np.round(start_times * sampling_frequency).astype(np.int64)
    valid_periods["end_sample_index"] = np.round(stop_times * sampling_frequency).astype(np.int64)
    valid_periods["unit_index"] = unit_indices

    ext_class = get_extension_class("valid_unit_periods")
    ext = ext_class(sorting_analyzer)
    ext.set_params(method="user_defined", user_defined_periods=valid_periods, minimum_valid_period_duration=0)
    ext.data["valid_unit_periods"] = valid_periods
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["valid_unit_periods"] = ext


def _load_pca_projections(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    """Dispatch to per-channel or concatenated PCA loader."""
    by_channel_nwb = getattr(extensions, "pca_projections_by_channel", None)
    concatenated_nwb = getattr(extensions, "pca_projections_concatenated", None)

    if by_channel_nwb is not None:
        _load_pca_by_channel(by_channel_nwb, sorting_analyzer)
    elif concatenated_nwb is not None:
        _load_pca_concatenated(concatenated_nwb, sorting_analyzer)


def _load_pca_by_channel(pc_nwb, sorting_analyzer: "SortingAnalyzer") -> None:
    """Load per-channel PCA projections (double-ragged) into the SortingAnalyzer."""
    from spikeinterface.core.sortinganalyzer import get_extension_class

    pc_data = pc_nwb.data.data[:]
    data_index = pc_nwb.data_index.data[:]
    data_index_index = pc_nwb.data_index_index.data[:]
    num_units = len(data_index_index)
    num_components = pc_data.shape[1]

    per_unit_projections = []
    max_sparse_channels = 0
    for unit_idx in range(num_units):
        spike_start = 0 if unit_idx == 0 else int(data_index_index[unit_idx - 1])
        spike_end = int(data_index_index[unit_idx])

        unit_projections = []
        for spike_idx in range(spike_start, spike_end):
            row_start = 0 if spike_idx == 0 else int(data_index[spike_idx - 1])
            row_end = int(data_index[spike_idx])
            n_channels = row_end - row_start
            max_sparse_channels = max(max_sparse_channels, n_channels)
            spike_pc = pc_data[row_start:row_end, :].T
            unit_projections.append(spike_pc)

        if unit_projections:
            per_unit_projections.append(np.stack(unit_projections))
        else:
            per_unit_projections.append(np.empty((0, num_components, 0), dtype=pc_data.dtype))

    some_spikes = sorting_analyzer.get_extension("random_spikes").get_random_spikes()
    total_spikes = len(some_spikes)
    all_projections = np.zeros((total_spikes, num_components, max_sparse_channels), dtype=np.float64)

    for unit_idx in range(num_units):
        spike_mask = some_spikes["unit_index"] == unit_idx
        unit_pcs = per_unit_projections[unit_idx]
        if unit_pcs.shape[0] > 0:
            n_ch = unit_pcs.shape[2]
            all_projections[spike_mask, :, :n_ch] = unit_pcs

    ext_class = get_extension_class("principal_components")
    ext = ext_class(sorting_analyzer)
    ext.set_params(n_components=num_components, mode="by_channel_local")
    ext.data["pca_projection"] = all_projections
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["principal_components"] = ext


def _load_pca_concatenated(pc_nwb, sorting_analyzer: "SortingAnalyzer") -> None:
    """Load concatenated PCA projections (single-ragged) into the SortingAnalyzer."""
    from spikeinterface.core.sortinganalyzer import get_extension_class

    pc_data = pc_nwb.data.data[:]
    data_index = pc_nwb.data_index.data[:]
    num_units = len(data_index)
    num_components = pc_data.shape[1]

    per_unit_projections = []
    for unit_idx in range(num_units):
        start = 0 if unit_idx == 0 else int(data_index[unit_idx - 1])
        end = int(data_index[unit_idx])
        per_unit_projections.append(pc_data[start:end])

    some_spikes = sorting_analyzer.get_extension("random_spikes").get_random_spikes()
    total_spikes = len(some_spikes)
    all_projections = np.zeros((total_spikes, num_components), dtype=np.float64)

    for unit_idx in range(num_units):
        spike_mask = some_spikes["unit_index"] == unit_idx
        unit_pcs = per_unit_projections[unit_idx]
        if unit_pcs.shape[0] > 0:
            all_projections[spike_mask] = unit_pcs

    ext_class = get_extension_class("principal_components")
    ext = ext_class(sorting_analyzer)
    ext.set_params(n_components=num_components, mode="concatenated")
    ext.data["pca_projection"] = all_projections
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["principal_components"] = ext
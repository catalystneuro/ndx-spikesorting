from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from hdmf.common import DynamicTable, VectorData, VectorIndex, DynamicTableRegion
from pynwb import NWBFile
from pynwb import get_type_map
from pynwb.ecephys import ElectricalSeries
from pynwb.epoch import TimeIntervals

if TYPE_CHECKING:
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
    NumSpikes,
    UnitVectorData,
    UnitsMetrics,
    ValidUnitPeriods,
    SpikeSortingExtensions,
    SpikeSortingContainer,
)


def find_extension_metric_targets(col_name: str) -> tuple[str, ...]:
    """Return the SI extension(s) a given metric column belongs to, based on COLUMN_TO_EXTENSION."""
    from spikeinterface.metrics import ComputeQualityMetrics, ComputeSpikeTrainMetrics, ComputeTemplateMetrics

    extension_targets = ()
    for potential_target in [
        ComputeQualityMetrics,
        ComputeSpikeTrainMetrics,
        ComputeTemplateMetrics,
    ]:
        if col_name in potential_target.get_metric_columns():
            extension_targets += (potential_target.extension_name,)
    return extension_targets

# Registries mapping canonical metric names to typed VectorData classes.
# The writer uses col_cls=<class> when adding a column whose class appears here;
# other columns are written as plain VectorData with the name preserved.
# The registries are the single point of SpikeInterface-specific knowledge in the loader;
# the spec stays tool-agnostic.
UNITS_TYPED_COLUMNS = {
    "firing_rate": FiringRate,
    "num_spikes": NumSpikes,
}


def _columns_supporting_periods() -> set[str]:
    """Set of metric column names whose underlying SI metric respects ``periods``.

    Walks ``ComputeQualityMetrics`` and ``ComputeSpikeTrainMetrics`` metric
    lists and collects ``metric_columns.keys()`` for each metric class whose
    ``supports_periods`` flag is True. Template metrics are skipped because
    none of them respect periods (template shape is time-invariant).

    Result is used by the writer to decide which metric columns receive a
    ``time_support`` reference: only the columns whose values actually depend
    on the periods get a link. Columns whose values are computed ignoring
    ``periods`` (e.g. ``snr``, PCA-based separability metrics) get no link
    even when ``qm_ext.params["periods"]`` is set, because the periods didn't
    actually constrain those values.
    """
    from spikeinterface.metrics import ComputeQualityMetrics, ComputeTemplateMetrics, ComputeSpikeTrainMetrics
    cols: set[str] = set()

    for metric_ext_cls in [ComputeQualityMetrics, ComputeSpikeTrainMetrics, ComputeTemplateMetrics]:
        for m in metric_ext_cls.metric_list:
            if getattr(m, "supports_periods", False):
                cols.update(m.metric_columns.keys())
    return cols


def templates_to_dense(
    templates: Templates, num_channels: int, electrode_rows: np.ndarray | None = None
) -> np.ndarray:
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
    electrode_rows : numpy.ndarray or None, optional
        The electrodes-table rows of the analyzer's channels, in channel order
        (``SpikeSortingContainer.electrodes.data``). Used to map the rows stored in
        ``templates.electrodes`` back to channel positions. If *None*, the stored
        rows are taken as channel positions, which holds when the analyzer's
        channels are rows ``0..num_channels-1``.

    Returns
    -------
    numpy.ndarray
        Dense templates with shape ``(num_units, num_samples, num_channels)``,
        dtype float32.
    """
    sparse_data = templates.data.data[:]
    data_index = templates.data_index.data[:]
    electrode_indices = templates.electrodes.data[:]
    if electrode_rows is not None:
        electrode_indices = _electrode_rows_to_channel_positions(electrode_indices, electrode_rows)

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


def _electrode_rows_to_channel_positions(rows, electrode_rows) -> np.ndarray:
    """Map electrodes-table rows to positions in ``electrode_rows`` (the analyzer's channels)."""
    row_to_position = {int(row): position for position, row in enumerate(electrode_rows)}
    missing = sorted({int(row) for row in rows} - row_to_position.keys())
    if missing:
        raise ValueError(f"Electrodes-table rows {missing} are not among the sorting analyzer's electrodes.")
    return np.array([row_to_position[int(row)] for row in rows], dtype=np.int64)


def _channel_ids_to_electrode_rows(sorting_analyzer: "SortingAnalyzer", nwbfile: NWBFile) -> np.ndarray:
    """Find the electrodes-table row of each ``sorting_analyzer.channel_ids`` entry.

    Matches channel ids against the ``channel_name`` column (written by neuroconv).
    Without that column, the analyzer's channels are assumed to be rows
    ``0..num_channels-1``, which only holds when the table has exactly that many rows.
    """
    electrodes = nwbfile.electrodes
    num_channels = sorting_analyzer.get_num_channels()
    if electrodes is None:
        raise ValueError("nwbfile has no electrodes table. Add electrodes before calling this function.")

    if "channel_name" not in electrodes.colnames:
        if len(electrodes) != num_channels:
            raise ValueError(
                f"The electrodes table has {len(electrodes)} rows and no 'channel_name' column, so its rows cannot "
                f"be matched to the sorting analyzer's {num_channels} channels. Pass `electrode_indices` explicitly."
            )
        return np.arange(num_channels, dtype=np.int64)

    channel_names = [str(name) for name in electrodes["channel_name"][:]]
    name_to_row = {name: row for row, name in enumerate(channel_names)}
    analyzer_names = [str(channel_id) for channel_id in sorting_analyzer.channel_ids]
    missing = [name for name in analyzer_names if name not in name_to_row]
    if missing:
        raise ValueError(
            f"sorting_analyzer.channel_ids {missing} have no matching 'channel_name' in the electrodes table. "
            "Pass `electrode_indices` explicitly."
        )
    ambiguous = sorted({name for name in analyzer_names if channel_names.count(name) > 1})
    if ambiguous:
        raise ValueError(
            f"Channel names {ambiguous} appear more than once in the electrodes table (e.g. one per probe), so "
            "they cannot be matched unambiguously. Pass `electrode_indices` explicitly."
        )
    return np.array([name_to_row[name] for name in analyzer_names], dtype=np.int64)


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
    electrode_rows: np.ndarray,
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
            all_wf_electrode_indices.extend(electrode_rows[channel_indices])
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
    electrode_rows: np.ndarray,
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
        all_electrode_indices.extend(electrode_rows[channel_indices])
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


### METRICS ZONE ###
def _add_cell_intrinsic_columns_to_units(
    sorting_analyzer: "SortingAnalyzer",
    units_table,
) -> set[str]:
    """Add canonical cell-intrinsic metric columns to ``units_table``.

    Pulls per-unit values from SI's ``metrics`` extension DataFrame and writes them
    as typed VectorData columns on the existing units table. Each column
    carries its type tag (e.g. ``FiringRate``) so a reader can recover the canonical
    mapping back to SI extensions.

    Analyzer units are matched to table rows by id, so the table may hold other
    units too; their rows get NaN (float metrics) or -1 (integer metrics). A column
    that already exists (e.g. written by another analyzer sharing the table) is left
    untouched, and its metric stays in ``UnitsMetrics`` instead.

    Returns the names of the columns that were written.
    """
    all_metrics = sorting_analyzer.get_metrics_extension_data()
    # remove dulpicate columns until SI 0.105.0 is released with a bugfix #4609
    all_metrics = all_metrics.loc[:, ~all_metrics.columns.duplicated()]
    ns_catalog = get_type_map().namespace_catalog
    row_indices = _analyzer_unit_positions_to_units_rows(sorting_analyzer, units_table)
    written = set()
    for metric_name, col_cls in UNITS_TYPED_COLUMNS.items():
        if metric_name in all_metrics.columns and metric_name not in units_table.colnames:
            spec = ns_catalog.get_spec("ndx-spikesorting", col_cls.__name__)
            analyzer_values = all_metrics[metric_name].to_numpy()
            fill_value = np.nan if np.issubdtype(analyzer_values.dtype, np.floating) else -1
            nwb_ordered = np.full(len(units_table), fill_value, dtype=analyzer_values.dtype)
            nwb_ordered[row_indices] = analyzer_values
            units_table.add_column(
                name=metric_name,
                description=spec.doc,
                data=nwb_ordered,
                col_cls=col_cls,
            )
            written.add(metric_name)
    return written


def _units_table_id_lookup(units_table) -> dict[str, int]:
    """Map each unit id in an NWB Units table to its row index.

    Prefers the ``unit_name`` column (canonical when string unit ids are
    written via neuroconv); falls back to the ``id`` ElementIdentifiers.
    Keys are stringified to match SpikeInterface's stringification on read.
    """
    if "unit_name" in units_table.colnames:
        ids = list(units_table["unit_name"][:])
    else:
        ids = list(units_table.id[:])
    return {str(uid): i for i, uid in enumerate(ids)}


def _analyzer_unit_positions_to_units_rows(sorting_analyzer, units_table) -> list[int]:
    """For each analyzer position i, return the nwbfile.units row index for that unit id.

    Raises ValueError if an analyzer unit id has no matching row in the units
    table. This is what makes the writer order-independent: the resulting list,
    used as DTR data, points each row of UnitsMetrics at the right nwbfile.units
    row regardless of whether the two tables share an ordering.
    """
    id_lookup = _units_table_id_lookup(units_table)
    analyzer_ids = [str(uid) for uid in sorting_analyzer.unit_ids]
    missing = [uid for uid in analyzer_ids if uid not in id_lookup]
    if missing:
        raise ValueError(
            f"sorting_analyzer.unit_ids has no matching row in nwbfile.units for: {missing}. "
            f"Available units in nwbfile.units: {sorted(id_lookup.keys())}"
        )
    return [id_lookup[uid] for uid in analyzer_ids]


def _build_periods_table(
    *,
    periods,
    sorting_analyzer: "SortingAnalyzer",
    units_table,
    table_cls,
    name: str,
    description: str,
):
    """Build a TimeIntervals-shaped table (rows = (start, stop, unit DTR)) from periods.

    Used to construct both ValidUnitPeriods instances (table_cls=ValidUnitPeriods)
    and plain TimeIntervals instances (table_cls=TimeIntervals) from SI's
    structured periods arrays.
    """
    if sorting_analyzer.get_num_segments() > 1:
        raise NotImplementedError(
            "Periods round-trip is single-segment only; "
            "multi-segment support not yet implemented."
        )
    sampling_frequency = sorting_analyzer.sampling_frequency
    analyzer_pos_to_row = _analyzer_unit_positions_to_units_rows(sorting_analyzer, units_table)

    start_times = [float(p["start_sample_index"]) / sampling_frequency for p in periods]
    stop_times = [float(p["end_sample_index"]) / sampling_frequency for p in periods]
    unit_row_indices = [analyzer_pos_to_row[int(p["unit_index"])] for p in periods]

    table = table_cls(name=name, description=description)
    # ValidUnitPeriods declares `unit` as a required column in its spec, so it's
    # already on the table after construction. Plain TimeIntervals doesn't; we
    # add it ourselves. Either way, add_row needs `unit` to exist before rows
    # are appended (add_row validates every existing column receives a value).
    if "unit" not in table.colnames:
        table.add_column(
            name="unit",
            description="Reference to units table for each row.",
            table=units_table,
        )
    else:
        # Bind the existing typed `unit` column to nwbfile.units. The empty
        # DTR doesn't carry a table reference until we either pass `table=` to
        # the constructor (which doesn't fit our deferred construction here)
        # or set it explicitly.
        table["unit"].table = units_table
    for start, stop, unit_row in zip(start_times, stop_times, unit_row_indices):
        table.add_row(start_time=start, stop_time=stop, unit=unit_row)
    return table


# TODO: simplify this
def _build_valid_unit_periods_and_qm_time_support(
    sorting_analyzer: "SortingAnalyzer",
    units_table=None,
):
    """Build the (ValidUnitPeriods, qm_time_support) pair for one analyzer.

    Two distinct interval sources may exist on a SortingAnalyzer:

    1. The ``valid_unit_periods`` SI extension's output. Written as a
       ``ValidUnitPeriods`` instance.
    2. ``quality_metrics.params["periods"]``. Written as a plain
       ``TimeIntervals`` instance with a ``unit`` DTR.

    When both exist with equal data (strict ``np.array_equal`` on the structured
    array), only the ValidUnitPeriods is written; metric columns link to it.
    When they differ, both are written; metric columns link to the qm-derived
    table because that's what the metrics were actually computed over.

    Returns a tuple ``(vup, qm_ts, ts_ref)``:
    - ``vup``: ValidUnitPeriods or None.
    - ``qm_ts``: plain TimeIntervals (only when qm periods differ from vup) or None.
    - ``ts_ref``: the table to link from MetricVectorData.time_support, or None.
    """
    vup_ext = sorting_analyzer.get_extension("valid_unit_periods")
    qm_ext = sorting_analyzer.get_extension("quality_metrics")

    vup_periods = vup_ext.get_data(outputs="numpy") if vup_ext is not None else None
    if vup_periods is not None and len(vup_periods) == 0:
        vup_periods = None

    qm_periods = qm_ext.params.get("periods") if qm_ext is not None and qm_ext.params else None
    if qm_periods is not None and len(qm_periods) == 0:
        qm_periods = None

    vup = None
    qm_ts = None
    ts_ref = None

    if vup_periods is not None:
        vup = _build_periods_table(
            periods=vup_periods,
            sorting_analyzer=sorting_analyzer,
            units_table=units_table,
            table_cls=ValidUnitPeriods,
            name="valid_unit_periods",
            description=(
                "Valid time periods per unit from spike sorting quality estimation."
            ),
        )

    if qm_periods is not None:
        sources_equal = (
            vup_periods is not None
            and qm_periods.dtype == vup_periods.dtype
            and np.array_equal(qm_periods, vup_periods)
        )
        if sources_equal:
            # qm and SI ext data are identical; reuse the single ValidUnitPeriods
            # as the time-support target.
            ts_ref = vup
        else:
            # qm periods differ from (or exist without) the SI extension data.
            # Write a plain TimeIntervals dedicated to the qm time support.
            qm_ts = _build_periods_table(
                periods=qm_periods,
                sorting_analyzer=sorting_analyzer,
                units_table=units_table,
                table_cls=TimeIntervals,
                name="unit_metrics_time_support",
                description=(
                    "Per-unit time intervals over which some of the unit_metrics values were "
                    "computed. Sourced from quality_metrics.params['periods']."
                ),
            )
            ts_ref = qm_ts

    return vup, qm_ts, ts_ref


def _convert_units_metrics(
    sorting_analyzer: "SortingAnalyzer",
    units_table,
    time_support_ref=None,
    skip_columns=frozenset(),
) -> UnitsMetrics | None:
    """Build a UnitsMetrics instance from SI's ``metrics`` extension.

    Each row of the table references one unit (via the ``unit``
    ``DynamicTableRegion``) and carries that unit's run-dependent metric values
    as ``UnitVectorData`` columns. When ``time_support_ref`` is provided, each
    metric column whose underlying SI metric has ``supports_periods=True`` gets
    a ``time_support`` attribute pointing to that table. Columns whose metric
    ignores periods (e.g. ``snr``) are left without a link, since the periods
    didn't actually constrain their values. Returns ``None`` when no metrics are computed.
    """
    df = sorting_analyzer.get_metrics_extension_data()

    # We need this until SI 0.105.0 is release (with a bugfix #4609)
    df = df.loc[:, ~df.columns.duplicated()]

    # Per-row unit DTR pointing back to nwbfile.units (one row per unit). We
    # resolve each analyzer unit_id to a nwbfile.units row index by identity
    # rather than position, so the DTR stays correct even if nwbfile.units is
    # reordered or interleaved with units we don't own.
    dtr_data = _analyzer_unit_positions_to_units_rows(sorting_analyzer, units_table)
    unit_column = DynamicTableRegion(
        name="unit",
        data=dtr_data,
        description="Reference to the units table row this metric value applies to.",
        table=units_table,
    )

    columns = [unit_column]

    # Cache the set of column names whose underlying metric respects periods.
    # We only attach the time_support reference to those; columns whose metric
    # ignores periods get no link even when a time support table exists.
    period_supporting_cols = _columns_supporting_periods() if time_support_ref is not None else set()

    # Each metric column becomes a UnitVectorData so it can carry the optional
    # time_support reference.
    for col_name in df.columns:
        if col_name in skip_columns:
            continue  # Skip columns that were promoted to typed VectorData on the units table
        col_values = df[col_name].to_numpy()
        col_kwargs = dict(
            name=col_name,
            data=col_values,
            description=f"Run-dependent metric: {col_name}.",
        )
        if time_support_ref is not None and col_name in period_supporting_cols:
            col_kwargs["time_support"] = time_support_ref
        columns.append(UnitVectorData(**col_kwargs))

    return UnitsMetrics(
        name="unit_metrics",
        description="Run-dependent per-unit metrics from one analysis run.",
        columns=columns,
    )


def _convert_pca_by_channel(
    sorting_analyzer: "SortingAnalyzer",
    nwbfile: NWBFile,
    nwb_waveforms: Waveforms,
    electrode_rows: np.ndarray,
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
            all_pc_electrode_indices.extend(electrode_rows[channel_indices])
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


def _convert_pca(sorting_analyzer, nwbfile, nwb_waveforms, electrode_rows):
    """Convert PCA extension, returning a dict with the correct attribute key."""
    pc_ext = sorting_analyzer.get_extension("principal_components")
    pc_mode = pc_ext.params.get("mode", "by_channel_local")
    if pc_mode in ("by_channel_local", "by_channel_global"):
        nwb_pca = _convert_pca_by_channel(sorting_analyzer, nwbfile, nwb_waveforms, electrode_rows)
        return {"pca_projections_by_channel": nwb_pca}
    return {"pca_projections_concatenated": _convert_pca_concatenated(sorting_analyzer, nwb_waveforms)}


def _convert_all_extensions(sorting_analyzer, nwbfile, electrode_rows, units_table, typed_unit_columns):  # noqa: C901
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
        converted["templates"] = _convert_templates(sorting_analyzer, nwbfile, electrode_rows)
    nbefore = templates_ext.nbefore if templates_ext is not None else None

    if _has("waveforms"):
        if "random_spikes" not in converted:
            raise ValueError("Waveforms extension requires random_spikes to be computed.")
        if nbefore is None:
            raise ValueError("Waveforms extension requires templates to be computed (for nbefore).")
        converted["waveforms"] = _convert_waveforms(
            sorting_analyzer, nwbfile, converted["random_spikes"], nbefore, electrode_rows
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
        converted.update(_convert_pca(sorting_analyzer, nwbfile, converted["waveforms"], electrode_rows))

    # Build interval tables: a ValidUnitPeriods if the SI valid_unit_periods
    # extension exists, and/or a plain TimeIntervals if quality_metrics has its
    # own periods that differ from the SI extension's data. The third return is
    # whichever table should serve as the time_support reference for period-
    # supporting metric columns (the qm-derived TimeIntervals when periods
    # diverge, the ValidUnitPeriods when they coincide or only it exists).
    vup, qm_ts, ts_ref = _build_valid_unit_periods_and_qm_time_support(
        sorting_analyzer, units_table=units_table
    )
    if vup is not None:
        converted["__valid_unit_periods__"] = vup
    if qm_ts is not None:
        converted["__time_intervals__"] = qm_ts

    # Run-dependent metrics flow into a UnitsMetrics instance (or several, as
    # support for multiple curation runs is added). For v1 there is one
    # UnitsMetrics per file, sourced from SI's quality_metrics extension.
    units_metrics = _convert_units_metrics(
        sorting_analyzer, units_table, time_support_ref=ts_ref, skip_columns=typed_unit_columns
    )
    if units_metrics is not None:
        converted["__units_metrics__"] = units_metrics

    return converted


def _build_extensions(sorting_analyzer, nwbfile, electrode_rows, units_table):
    """Convert extensions and assemble a SpikeSortingExtensions object.

    Cell-intrinsic metric columns are added directly to ``units_table``
    (a side effect) rather than to the extensions container, because they
    are per-unit properties of the cells themselves.
    """
    # Side effect: write cell-intrinsic metric columns onto the units table.
    typed_unit_columns = _add_cell_intrinsic_columns_to_units(sorting_analyzer, units_table)

    converted = _convert_all_extensions(sorting_analyzer, nwbfile, electrode_rows, units_table, typed_unit_columns)
    extensions = SpikeSortingExtensions(name="extensions")
    for attr, obj in converted.items():
        if attr == "__units_metrics__":
            extensions.add_units_metrics(obj)
        elif attr == "__time_intervals__":
            extensions.add_time_intervals(obj)
        elif attr == "__valid_unit_periods__":
            extensions.add_valid_unit_periods(obj)
        else:
            setattr(extensions, attr, obj)
    return extensions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_sorting_analyzer_to_nwbfile(
    sorting_analyzer: "SortingAnalyzer",
    nwbfile: NWBFile,
    electrical_series: ElectricalSeries | None = None,
    units_table: DynamicTable | None = None,
    electrode_indices: list[int] | np.ndarray | None = None,
    container_name: str = "spike_sorting",
):
    """Add sorting analyzer extensions to an NWBFile.

    Converts computed SpikeInterface extensions from a ``SortingAnalyzer``
    into ndx-spikesorting NWB types and adds them inside a
    ``SpikeSortingContainer`` under the ``ecephys`` processing module.

    The *nwbfile* must already contain an electrodes table and a units table
    (e.g. added via ``neuroconv``). Both tables may hold rows that do not belong
    to this analyzer (other probes, other sortings): every region written here
    points at the rows of this analyzer's channels and units.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SpikeInterface ``SortingAnalyzer`` with one or more computed
        extensions.
    nwbfile : NWBFile
        The target NWB file. Must already have electrodes and units.
    electrical_series : ElectricalSeries or None, optional
        The ``ElectricalSeries`` the analyzer was computed on, linked as the container's
        source recording. It must cover all of the analyzer's electrodes and be sampled at
        the analyzer's sampling frequency. If *None* (default), no recording is linked, and
        ``read_sorting_analyzer_from_nwb`` returns a recordingless analyzer.
    units_table : DynamicTable or None, optional
        The units table holding the analyzer's units, matched by ``unit_name``
        (or ``id``). If *None*, ``nwbfile.units`` is used.
    electrode_indices : list of int or None, optional
        The electrodes-table row of each ``sorting_analyzer.channel_ids`` entry,
        in the same order. If *None*, rows are matched by the ``channel_name``
        column of the electrodes table.
    container_name : str, default: "spike_sorting"
        Name of the ``SpikeSortingContainer``. Use a distinct name for each
        analyzer written to the same file.
    """
    if sorting_analyzer.get_num_segments() > 1:
        raise NotImplementedError("Currently only single-segment SortingAnalyzers are supported.")

    # -- resolve the container location --
    if "ecephys" in nwbfile.processing:
        ecephys_module = nwbfile.processing["ecephys"]
    else:
        ecephys_module = nwbfile.create_processing_module(
            name="ecephys",
            description="Extracellular electrophysiology processing results",
        )
    if container_name in ecephys_module.data_interfaces:
        raise ValueError(
            f"processing/ecephys already holds an object named '{container_name}'. "
            "Pass a different `container_name` for each analyzer written to the same file."
        )

    # -- resolve electrodes rows --
    num_channels = sorting_analyzer.get_num_channels()
    if electrode_indices is None:
        electrode_rows = _channel_ids_to_electrode_rows(sorting_analyzer, nwbfile)
    else:
        electrode_rows = np.asarray(electrode_indices, dtype=np.int64)
        if len(electrode_rows) != num_channels:
            raise ValueError(
                f"`electrode_indices` has {len(electrode_rows)} entries but the sorting analyzer has "
                f"{num_channels} channels."
            )

    # -- resolve and validate the linked electrical series --
    if electrical_series is not None:
        missing_rows = sorted(set(electrode_rows.tolist()) - set(int(r) for r in electrical_series.electrodes.data[:]))
        if missing_rows:
            raise ValueError(
                f"ElectricalSeries '{electrical_series.name}' does not cover the sorting analyzer's electrodes rows "
                f"{missing_rows}. Pass the ElectricalSeries of the recording the analyzer was computed on."
            )
        if electrical_series.rate is not None and not np.isclose(
            electrical_series.rate, sorting_analyzer.sampling_frequency
        ):
            raise ValueError(
                f"ElectricalSeries '{electrical_series.name}' is sampled at {electrical_series.rate} Hz but the "
                f"sorting analyzer at {sorting_analyzer.sampling_frequency} Hz (e.g. an LFP series). Pass the "
                "ElectricalSeries of the recording the analyzer was computed on."
            )

    # -- resolve units table --
    if units_table is None:
        units_table = nwbfile.units
    if units_table is None:
        raise ValueError("nwbfile has no units table. Add units before calling this function.")

    # -- build electrode & unit regions --
    electrodes_region = nwbfile.create_electrode_table_region(
        region=electrode_rows.tolist(),
        description="All electrodes used in sorting analysis",
    )
    units_region = DynamicTableRegion(
        name="units_region",
        data=_analyzer_unit_positions_to_units_rows(sorting_analyzer, units_table),
        description="All units from sorting analysis",
        table=units_table,
    )

    # -- convert & assemble extensions --
    extensions = _build_extensions(sorting_analyzer, nwbfile, electrode_rows, units_table)

    # -- assemble SpikeSortingContainer --
    sparsity = sorting_analyzer.sparsity
    sparsity_mask = sparsity.mask if sparsity is not None else None

    container_kwargs = dict(
        name=container_name,
        sampling_frequency=sorting_analyzer.sampling_frequency,
        num_samples=np.int64(sorting_analyzer.get_num_samples()),
        electrodes=electrodes_region,
        units_region=units_region,
        sparsity_mask=sparsity_mask,
    )
    if electrical_series is not None:
        container_kwargs["source_electrical_series"] = electrical_series

    container = SpikeSortingContainer(**container_kwargs)
    container.spike_sorting_extensions = extensions
    ecephys_module.add(container)


def _nwb_object_path(obj) -> str:
    """Path of a units table or ElectricalSeries read from a file, as SpikeInterface's NWB extractors look it up.

    Same key as ``spikeinterface.extractors.nwbextractors._retrieve_unit_table_pynwb`` and
    ``_retrieve_electrical_series_pynwb``: the dataset's path in the file, which only exists once it is written.
    """
    dataset, suffix = (obj.id.data, "/id") if isinstance(obj, DynamicTable) else (obj.data, "/data")
    return dataset.name.removesuffix(suffix).lstrip("/")


def _rec_attributes_from_container(container, nwbfile, sorting) -> dict:
    """The ``rec_attributes`` of a recordingless analyzer, built from the container and the electrodes table.

    Channel ids come from the ``channel_name`` column (or the rows themselves), and the probe from
    ``rel_x``/``rel_y`` when present.
    """
    from probeinterface import Probe, ProbeGroup

    electrodes = nwbfile.electrodes
    rows = np.asarray(container.electrodes.data[:], dtype=np.int64)
    if "channel_name" in electrodes.colnames:
        channel_ids = np.array([str(electrodes["channel_name"][row]) for row in rows])
    else:
        channel_ids = rows

    probegroup = None
    if {"rel_x", "rel_y"} <= set(electrodes.colnames):
        # rel_x/rel_y are relative to the electrode group, so each group is its own probe (as NwbRecordingExtractor).
        positions = np.column_stack([np.asarray(electrodes[column].data[:])[rows] for column in ("rel_x", "rel_y")])
        group_names = np.array([str(electrodes["group_name"][row]) for row in rows])
        probegroup = ProbeGroup()
        for group_name in dict.fromkeys(group_names):
            (indices,) = np.nonzero(group_names == group_name)
            probe = Probe(ndim=2)
            probe.set_contacts(positions=positions[indices])
            probe.set_device_channel_indices(indices)
            probegroup.add_probe(probe)

    num_samples = container.num_samples
    return dict(
        channel_ids=channel_ids,
        sampling_frequency=container.sampling_frequency,
        num_channels=len(rows),
        num_samples=[num_samples],
        is_filtered=True,
        properties={},
        dtype=np.dtype("float32"),
        probegroup=probegroup,
    )


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
    from spikeinterface.core import ChannelSparsity, NumpySorting, SortingAnalyzer, create_sorting_analyzer
    from spikeinterface.extractors.nwbextractors import NwbRecordingExtractor, NwbSortingExtractor

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

        # -- Load sorting, restricted to the units this container references --
        units_region = container.units_region
        units_table = units_region.table
        sorting = NwbSortingExtractor(
            file_path=nwbfile_path,
            sampling_frequency=sampling_frequency,
            t_start=0.0,
            unit_table_path=_nwb_object_path(units_table),
        )
        row_to_id = {row: uid for uid, row in _units_table_id_lookup(units_table).items()}
        str_id_to_sorting_id = {str(uid): uid for uid in sorting.unit_ids}
        container_unit_ids = [str_id_to_sorting_id[row_to_id[int(row)]] for row in units_region.data[:]]
        if list(sorting.unit_ids) != container_unit_ids:
            sorting = sorting.select_units(container_unit_ids)

        # -- Load recording (if linked), restricted to the analyzer's electrodes --
        electrode_rows = np.asarray(container.electrodes.data[:], dtype=np.int64)
        recording = None
        if container.source_electrical_series is not None:
            es = container.source_electrical_series
            recording = NwbRecordingExtractor(
                file_path=nwbfile_path, electrical_series_path=_nwb_object_path(es)
            )
            es_rows = np.asarray(es.electrodes.data[:], dtype=np.int64)
            if not np.array_equal(es_rows, electrode_rows):
                positions = _electrode_rows_to_channel_positions(electrode_rows, es_rows)
                recording = recording.select_channels(recording.channel_ids[positions])

        # -- Without a linked recording, the analyzer is recordingless and its channels come from the container --
        rec_attributes = None
        if recording is None:
            rec_attributes = _rec_attributes_from_container(container, nwbfile, sorting)

        # -- Build sparsity from stored mask --
        sparsity = None
        if container.sparsity_mask is not None:
            channel_ids = recording.channel_ids if recording is not None else rec_attributes["channel_ids"]
            sparsity = ChannelSparsity(
                mask=np.array(container.sparsity_mask[:], dtype=bool),
                unit_ids=sorting.unit_ids,
                channel_ids=channel_ids,
            )

        # -- Create in-memory SortingAnalyzer --
        if recording is not None:
            sorting_analyzer = create_sorting_analyzer(
                sorting=sorting,
                recording=recording,
                format="memory",
                sparse=False,
                sparsity=sparsity,
            )
        else:
            sorting_analyzer = SortingAnalyzer(
                sorting=NumpySorting.from_sorting(sorting, with_metadata=True, copy_spike_vector=True),
                recording=None,
                rec_attributes=rec_attributes,
                format="memory",
                sparsity=sparsity,
            )

        # -- Instantiate precomputed extensions --
        extensions = container.spike_sorting_extensions
        if extensions is not None:
            _load_random_spikes(extensions, sorting_analyzer)
            _load_waveforms(extensions, sorting_analyzer)
            _load_templates(extensions, sorting_analyzer, electrode_rows)
            _load_noise_levels(extensions, sorting_analyzer)
            _load_unit_locations(extensions, sorting_analyzer)
            _load_correlograms(extensions, sorting_analyzer)
            _load_isi_histograms(extensions, sorting_analyzer)
            _load_template_similarity(extensions, sorting_analyzer)
            _load_spike_amplitudes(extensions, sorting_analyzer)
            _load_amplitude_scalings(extensions, sorting_analyzer)
            _load_spike_locations(extensions, sorting_analyzer)
            _load_pca_projections(extensions, sorting_analyzer)
            _load_units_metrics(extensions, sorting_analyzer)
            _load_valid_unit_periods_from_nwb(extensions, sorting_analyzer)

        # Cell-intrinsic metrics live on the units table as typed columns; read
        # them by column type so the SI extension binding survives even if a
        # column was renamed.
        _load_cell_intrinsic_from_units(units_region, sorting_analyzer)

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
    # Set directly: spikeinterface's `set_params` reads the recording even when `dtype` is given, which a
    # recordingless analyzer does not have. The window comes from the stored peak index, as for templates.
    peak_sample_index = waveforms_nwb.peak_sample_index
    ext.params = dict(
        ms_before=float(peak_sample_index / sorting_analyzer.sampling_frequency * 1000.0),
        ms_after=float((num_samples - peak_sample_index) / sorting_analyzer.sampling_frequency * 1000.0),
        dtype=all_waveforms.dtype.str,
    )
    ext.data["waveforms"] = all_waveforms
    ext.run_info = {"run_completed": True, "runtime_s": 0.0}
    sorting_analyzer.extensions["waveforms"] = ext


def _load_templates(extensions, sorting_analyzer: "SortingAnalyzer", electrode_rows=None) -> None:
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

    dense_templates = templates_to_dense(templates_nwb, num_channels, electrode_rows)

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


def _load_cell_intrinsic_from_units(units_region, sorting_analyzer: "SortingAnalyzer") -> None:
    """Read canonical cell-intrinsic typed columns from the rows of a Units table that ``units_region`` references.

    Walks the Units table columns, finds those that are instances of canonical
    typed VectorData classes, and populates the corresponding SI extension
    DataFrames. Routes by column class via ``UNITS_TYPED_COLUMNS``.

    When a target SI extension was already populated by ``_load_units_metrics``
    (e.g. quality_metrics receives both ``Snr`` from a UnitsMetrics and
    ``AmplitudeMedian`` from a Units column), this function merges into the
    existing DataFrame rather than overwriting it.
    """
    import pandas as pd
    from spikeinterface.core.sortinganalyzer import get_extension_class

    # Only the rows this container references belong to the analyzer; map each
    # of them to its analyzer unit position so values land on the right units.
    units_table = units_region.table
    rows = np.asarray(units_region.data[:], dtype=np.int64)
    analyzer_positions = _dtr_rows_to_analyzer_positions(units_region, sorting_analyzer)

    for col_name in units_table.colnames:
        col = units_table[col_name]
        for metric_name, col_cls in UNITS_TYPED_COLUMNS.items():
            if isinstance(col, col_cls):
                raw_values = np.asarray(col.data[:])
                if raw_values.ndim > 1:
                    raw_values = raw_values[:, 0]
                reordered_values = np.empty(len(analyzer_positions), dtype=raw_values.dtype)
                reordered_values[analyzer_positions] = raw_values[rows]
                target_extensions = find_extension_metric_targets(metric_name)
                for si_ext_name in target_extensions:
                    metric_ext = sorting_analyzer.extensions.get(si_ext_name)
                    if metric_ext is not None:
                        # If the extension is already present (e.g. from loading a UnitsMetrics), 
                        # merge this column into it rather than overwriting it.
                        existing_df = metric_ext.get_data()
                        if metric_name not in existing_df:
                            existing_df[metric_name] = reordered_values
                            metric_ext.params.setdefault("metric_names", []).append(metric_name)
                            metric_ext.params["metrics_to_compute"] = list(metric_ext.params["metric_names"])
                    else:
                        # If the extension is not available, make it from scratch so the metric can be added to it.
                        # This allows cell-intrinsic metrics to be loaded even when their extension's UnitsMetrics 
                        # counterpart is missing.
                        ext_class = get_extension_class(si_ext_name)
                        unit_ids = sorting_analyzer.unit_ids
                        metrics_df = pd.DataFrame(data={metric_name: reordered_values}, index=unit_ids)
                        ext = ext_class(sorting_analyzer)
                        ext.params = {
                            "metric_names": [metric_name],
                            "metric_params": {},
                            "metrics_to_compute": [metric_name],
                            "delete_existing_metrics": False,
                        }
                        ext.data["metrics"] = metrics_df
                        ext.run_info = {"run_completed": True, "runtime_s": 0.0}
                        sorting_analyzer.extensions[si_ext_name] = ext


def _collect_units_metrics_columns(nwb_table, sorting_analyzer):
    """Walk a UnitsMetrics table once and return (per_extension_columns, time_support_link).

    Reorders each column's row values into analyzer-unit-position order using
    the DTR-resolved row → analyzer position map, so the resulting arrays are
    aligned with ``sorting_analyzer.unit_ids`` regardless of whether the on-disk
    row order matches.
    """
    row_to_analyzer_pos = _dtr_rows_to_analyzer_positions(nwb_table["unit"], sorting_analyzer)
    # Permutation that places UnitsMetrics row i's value at the slot corresponding
    # to its analyzer unit position in the output array.
    reorder = np.argsort(row_to_analyzer_pos)

    per_extension_columns: dict[str, dict[str, np.ndarray]] = {}
    time_support_link = None
    for col_name in nwb_table.colnames:
        if col_name == "unit":
            continue
        col = nwb_table[col_name]
        if time_support_link is None:
            time_support_link = getattr(col, "time_support", None)

        si_ext_targets = find_extension_metric_targets(col_name)
        values = np.asarray(col.data[:])[reorder]
        for si_ext_name in si_ext_targets:
            per_extension_columns.setdefault(si_ext_name, {})[col_name] = values
    return per_extension_columns, time_support_link


def _load_units_metrics(extensions, sorting_analyzer: "SortingAnalyzer") -> None:
    """Read UnitsMetrics instances and reconstruct the relevant SI extensions.

    Each UnitsMetrics row holds one unit's metric values; column names identify
    the metric. The loader routes each column to the SI extension it belongs to
    using ``find_extension_metric_targets``. A column listed under multiple extensions
    (e.g., ``num_spikes`` belongs to both ``quality_metrics`` and
    ``spiketrain_metrics``) is replicated into each target. A column not in the
    dict falls back to ``quality_metrics`` (matching pre-fix behavior) and is
    surfaced via a warning so the dict can be kept in sync with SI's registries.
    """
    import pandas as pd
    from spikeinterface.core.base import unit_period_dtype
    from spikeinterface.core.sortinganalyzer import get_extension_class

    units_metrics = getattr(extensions, "units_metrics", None)
    if not units_metrics:
        return

    # Iterate over UnitsMetrics instances (LabelledDict keyed by name)
    runs = units_metrics.values() if hasattr(units_metrics, "values") else [units_metrics]
    for nwb_table in runs:
        unit_ids = sorting_analyzer.unit_ids
        per_extension_params: dict[str, dict] = {}
        per_extension_columns, time_support_link = _collect_units_metrics_columns(
            nwb_table, sorting_analyzer
        )

        for si_ext_name, columns_dict in per_extension_columns.items():
            ext_class = get_extension_class(si_ext_name)
            if ext_class is None:
                continue
            existing = sorting_analyzer.extensions.get(si_ext_name)
            if existing is not None and existing.data.get("metrics") is not None:
                # Merge into an extension already populated by an earlier loader
                # step. Preserve existing metric_names; append new ones.
                existing_df = existing.data["metrics"]
                for col_name_inner, values in columns_dict.items():
                    existing_df[col_name_inner] = values
                metric_names = existing.params.setdefault("metric_names", [])
                for n in columns_dict:
                    if n not in metric_names:
                        metric_names.append(n)
                existing.params["metrics_to_compute"] = list(metric_names)
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

        # Restore quality_metrics.params["periods"] from the column-level
        # time_support reference, when present.
        qm_periods = _periods_from_time_support_link(time_support_link, sorting_analyzer, unit_period_dtype)
        qm_ext = sorting_analyzer.extensions.get("quality_metrics")
        if qm_periods is not None and qm_ext is not None:
            qm_ext.params["periods"] = qm_periods


def _dtr_rows_to_analyzer_positions(unit_dtr, sorting_analyzer) -> np.ndarray:
    """Map DTR row indices (into nwbfile.units) into analyzer unit positions.

    DTR ``data[i]`` is the nwbfile.units row index for the i-th referencing row.
    We resolve each row back to its unit id (preferring ``unit_name``) and look
    that id up in ``sorting_analyzer.unit_ids`` so the resulting integer is the
    correct positional index into the analyzer regardless of any reordering.
    """
    units_table = unit_dtr.table
    row_to_id = {row_idx: uid for uid, row_idx in _units_table_id_lookup(units_table).items()}
    analyzer_id_to_pos = {str(uid): pos for pos, uid in enumerate(sorting_analyzer.unit_ids)}
    dtr_rows = np.asarray(unit_dtr.data[:], dtype=np.int64)
    out = np.empty(len(dtr_rows), dtype=np.int64)
    for i, row_idx in enumerate(dtr_rows):
        uid = row_to_id[int(row_idx)]
        if uid not in analyzer_id_to_pos:
            raise ValueError(
                f"DTR references nwbfile.units row {int(row_idx)} (unit id {uid!r}), "
                f"but that id is not in sorting_analyzer.unit_ids: {list(analyzer_id_to_pos.keys())}"
            )
        out[i] = analyzer_id_to_pos[uid]
    return out


def _periods_from_time_support_link(vup_table, sorting_analyzer, unit_period_dtype):
    """Build SI's structured periods array from a linked ValidUnitPeriods table.

    ``vup_table`` is a ``ValidUnitPeriods`` (TimeIntervals subclass) referenced
    from a metric column's ``time_support`` attribute. Returns ``None`` when no
    link was set. Single-segment only (segment_index=0).
    """
    if vup_table is None:
        return None
    start_times = np.asarray(vup_table["start_time"][:], dtype=np.float64)
    stop_times = np.asarray(vup_table["stop_time"][:], dtype=np.float64)
    analyzer_positions = _dtr_rows_to_analyzer_positions(vup_table["unit"], sorting_analyzer)
    sampling_frequency = sorting_analyzer.sampling_frequency

    n_periods = len(start_times)
    periods = np.zeros(n_periods, dtype=unit_period_dtype)
    periods["segment_index"] = 0
    periods["start_sample_index"] = np.round(start_times * sampling_frequency).astype(np.int64)
    periods["end_sample_index"] = np.round(stop_times * sampling_frequency).astype(np.int64)
    periods["unit_index"] = analyzer_positions
    return periods


def _load_valid_unit_periods_from_nwb(extensions, sorting_analyzer):
    from spikeinterface.core.base import unit_period_dtype
    from spikeinterface.core.sortinganalyzer import get_extension_class

    # valid_unit_periods is now a LabelledDict (quantity="*"). For SI extension
    # restoration we pick the one named "valid_unit_periods" by convention; if
    # not present, the first instance (or none).
    valid_periods_dict = getattr(extensions, "valid_unit_periods", None)
    if not valid_periods_dict:
        return
    if hasattr(valid_periods_dict, "get") and "valid_unit_periods" in valid_periods_dict:
        valid_periods_nwb = valid_periods_dict["valid_unit_periods"]
    elif hasattr(valid_periods_dict, "values"):
        try:
            valid_periods_nwb = next(iter(valid_periods_dict.values()))
        except StopIteration:
            return
    else:
        valid_periods_nwb = valid_periods_dict

    start_times = np.array(valid_periods_nwb["start_time"][:], dtype=np.float64)
    stop_times = np.array(valid_periods_nwb["stop_time"][:], dtype=np.float64)
    analyzer_positions = _dtr_rows_to_analyzer_positions(valid_periods_nwb["unit"], sorting_analyzer)

    sampling_frequency = sorting_analyzer.sampling_frequency
    n_periods = len(start_times)

    valid_periods = np.zeros(n_periods, dtype=unit_period_dtype)
    valid_periods["segment_index"] = 0
    valid_periods["start_sample_index"] = np.round(start_times * sampling_frequency).astype(np.int64)
    valid_periods["end_sample_index"] = np.round(stop_times * sampling_frequency).astype(np.int64)
    valid_periods["unit_index"] = analyzer_positions

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
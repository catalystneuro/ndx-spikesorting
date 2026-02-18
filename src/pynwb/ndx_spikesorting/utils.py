import numpy as np
from ndx_spikesorting import Templates


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
    import numpy as np

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

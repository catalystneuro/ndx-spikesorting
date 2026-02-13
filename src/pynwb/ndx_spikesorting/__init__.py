from importlib.resources import files

from pynwb import load_namespaces, get_class, register_class

# Get path to the namespace.yaml file with the expected location when installed not in editable mode
__location_of_this_file = files(__name__)
__spec_path = __location_of_this_file / "spec" / "ndx-spikesorting.namespace.yaml"

# If that path does not exist, we are likely running in editable mode. Use the local path instead
if not __spec_path.exists():
    __spec_path = __location_of_this_file.parent.parent.parent / "spec" / "ndx-spikesorting.namespace.yaml"

# Load the namespace
load_namespaces(str(__spec_path))

# Auto-generate classes from the spec
RandomSpikes = get_class("RandomSpikes", "ndx-spikesorting")
AutoTemplates = get_class("Templates", "ndx-spikesorting")
SpikeSortingExtensions = get_class("SpikeSortingExtensions", "ndx-spikesorting")
SpikeSortingContainer = get_class("SpikeSortingContainer", "ndx-spikesorting")


@register_class(neurodata_type="Templates", namespace="ndx-spikesorting")
class Templates(AutoTemplates):
    """Template waveforms per unit stored as a ragged array."""

    def to_dense(self, num_channels):
        """Convert sparse ragged templates to a dense 3D array.

        Reconstructs a dense array of shape ``(num_units, num_samples, num_channels)``
        from the sparse ragged representation stored in NWB. Inactive channels are
        filled with zeros.

        Parameters
        ----------
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

        sparse_data = self.data.data[:]
        data_index = self.data_index.data[:]
        electrode_indices = self.electrodes.data[:]

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


__all__ = [
    "RandomSpikes",
    "Templates",
    "SpikeSortingExtensions",
    "SpikeSortingContainer",
]

# Remove these functions/modules from the package
del load_namespaces, get_class, register_class, files
del __location_of_this_file, __spec_path, AutoTemplates

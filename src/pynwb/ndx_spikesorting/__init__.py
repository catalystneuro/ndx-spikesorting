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
Templates = get_class("Templates", "ndx-spikesorting")
UnitLocations = get_class("UnitLocations", "ndx-spikesorting")
SpikeSortingExtensions = get_class("SpikeSortingExtensions", "ndx-spikesorting")
SpikeSortingContainer = get_class("SpikeSortingContainer", "ndx-spikesorting")

from .utils import templates_to_dense

__all__ = [
    "RandomSpikes",
    "Templates",
    "UnitLocations",
    "SpikeSortingExtensions",
    "SpikeSortingContainer",
    "templates_to_dense",
]

# Remove these functions/modules from the package
del load_namespaces, get_class, register_class, files
del __location_of_this_file, __spec_path

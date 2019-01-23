import importlib

from .. import config
from . import path
from . import visualize
from . import machine
from . import data

if not config.RELEASE:
    importlib.reload(path)
    importlib.reload(visualize)
    importlib.reload(machine)
    importlib.reload(data)
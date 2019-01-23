import importlib

from .. import config
from . import pipeline

if not config.RELEASE:
    importlib.reload(pipeline)

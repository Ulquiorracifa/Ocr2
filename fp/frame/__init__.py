import importlib

from .. import config
from . import surface
from . import textline
from . import wireframe
from . import template_data
from . import template
from . import table

importlib.reload(config)
if not config.RELEASE:
    importlib.reload(surface)
    importlib.reload(textline)
    importlib.reload(wireframe)
    importlib.reload(template_data)
    importlib.reload(template)
    importlib.reload(table)

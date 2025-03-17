# potentially import time utils
from ipd.bunch import Bunch as Bunch
from ipd.dev.contexts import *
from ipd.dev.error import *
from ipd.dev.instrumentation import *
from ipd.dev.instrumentation import timed as profile  # noqa

# runtime utils
from ipd.dev.debug import *
from ipd.dev.objinfo import *
from ipd.dev.iterables import *
from ipd.dev.format import *
from ipd.dev.code import *
from ipd.dev.decorators import *
from ipd.dev.element_wise import *
from ipd.dev.meta import *
from ipd.dev.metadata import *
from ipd.dev.misc import *
from ipd.observer import *
from ipd.dev.safe_eval import *
from ipd.dev.serialization import *
from ipd.dev.shell import *
from ipd.dev.state import *
from ipd.dev.storage import *
from ipd.dev.strings import *
from ipd.project_config import *
from ipd.dev.tolerances import *
from ipd.dev.testing import *

from ipd import lazyimport as lazyimport
# utils involving optional dependencies
cli = lazyimport('ipd.dev.cli')
cuda = lazyimport('ipd.dev.cuda')
qt = lazyimport('ipd.dev.qt')

_global_timer = None

def __getattr__(name) -> Any:
    if name == 'global_timer':
        global _global_timer
        return (_global_timer := _global_timer or Timer())
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from ipd.dev import LazyModule

server = LazyModule('ipd.ppp.server')

from ipd.ppp.models import *

for cls in client_models.values():
    globals()[cls.__name__] = cls

from ipd.ppp.client import *
from ipd.dev import timed as profile

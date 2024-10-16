from ipd.dev import LazyModule

server = LazyModule('ipd.ppp.server')

from ipd.ppp.models import *
import ipd.ppp.dbmodels as dbmodels
from ipd.ppp.dbmodels import client_models

for cls in client_models.values():
    globals()[cls.__name__] = cls

from ipd.ppp.client import *

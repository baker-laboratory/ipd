from ipd.dev import LazyModule

server = LazyModule('ipd.ppp.server')

from ipd.ppp.models import *
import ipd.ppp.dbmodels as dbmodels
from ipd.ppp.dbmodels import client_model

for cls in client_model.values():
    globals()[cls.__name__] = cls

from ipd.ppp.client import *

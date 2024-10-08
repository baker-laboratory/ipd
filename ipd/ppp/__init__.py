from ipd.dev import LazyModule

server = LazyModule('ipd.ppp.server')
from ipd.ppp.models import *
from ipd.ppp.client import *
from ipd.ppp.defaults import *

from ipd.dev import lazyimport

fastapi = lazyimport('fastapi', 'fastapi[standard]', pip=True).now()
it = lazyimport('itertools', 'more_itertools', pip=True).now()
ordset = lazyimport('ordered_set', pip=True).now()
pymol = lazyimport('pymol', 'pymol-bundle', mamba=True, channels='-c schrodinger').now()
requests = lazyimport('requests', pip=True).now()
rich = lazyimport('rich', 'Rich', pip=True).now()
sqlmodel = lazyimport('sqlmodel', pip=True).now()
wpc = lazyimport('wills_pymol_crap', 'git+https://github.com/willsheffler/wills_pymol_crap.git',
                 pip=True).now()
yaml = lazyimport('yaml', 'pyyaml', pip=True).now()

from ipd.dev import timed as profile  # noqa
from ipd.ppp.models import *
from ipd.ppp.client import *
from ipd.ppp.server import PPPBackend as PPPBackend

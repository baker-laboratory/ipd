from ipd.dev.bunch import *
from ipd.dev.lazy_import import *
from ipd.dev.safe_eval import *
from ipd.dev.timer import *
from ipd.dev.git import *
from ipd.dev.contexts import *

def fstr(template):
    return safe_eval(f'f"""{template}"""')

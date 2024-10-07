import io
import contextlib
from ipd.dev.contexts import *
from ipd.dev.safe_eval import *

def fstr(template):
    return safe_eval(f'f"""{template}"""')

def printed_string(thing):
    with contextlib.suppress(ImportError):
        from rich import print
    strio = io.StringIO()
    with redirect(stdout=strio, stderr='stdout'):
        print(thing)
        strio.seek(0)
        return strio.read()

from ipd.dev.bunch import *
from ipd.dev.lazy_import import *
from ipd.dev.timer import *
from ipd.dev.git import *
from ipd.dev.toggle import *
from ipd.dev.state import *
from ipd.dev.filefetch import *

import contextlib
import io
import re

from ipd.dev.contexts import redirect

def get_all_annotations(cls):
    annotations = {}
    for base in cls.__mro__[::-1]:
        annotations |= getattr(base, '__annotations__', {})
    return annotations

def eval_fstring(template, namespace):
    return eval(f'f"""{template}"""', namespace)

def printed_string(thing, rich=True):
    with contextlib.suppress(ImportError):
        if rich: from rich import print
    strio = io.StringIO()
    with redirect(stdout=strio, stderr='stdout'):
        print(thing)  # type: ignore
        strio.seek(0)
        return strio.read()

def strip_duplicate_spaces(s):
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s

def tobytes(s):
    if isinstance(s, str):
        return s.encode()
    return s

def tostr(s):
    if isinstance(s, bytes):
        return s.decode()
    return s

def toname(val):
    if not re.match(r'[%^&*#$]', val): return val
    return None

def toidentifier(val):
    if isinstance(val, str) and val.isidentifier(): return val
    return None

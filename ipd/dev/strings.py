import io

from ipd.dev.contexts import redirect
from ipd.dev.safe_eval import safe_eval

def get_all_annotations(cls):
    annotations = {}
    for base in cls.__mro__[::-1]:
        annotations |= getattr(base, '__annotations__', {})
    return annotations

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

def strip_duplicate_spaces(s):
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s

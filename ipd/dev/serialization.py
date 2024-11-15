import os
import json
import re
from pathlib import Path

fixlist = re.compile(r'(^|[^"])(\[[^"\[\]\{\}\(\)]+\])($|[^"])')

def str_to_json_recurse(val):
    if isinstance(val, dict): return {k: str_to_json_recurse(v) for k, v in val.items()}
    if isinstance(val, list): return [str_to_json_recurse(v) for v in val]
    if isinstance(val, Path): return str(val)
    if isinstance(val, str) and val:
        if (val[0], val[-1]) == ('[', ']'):
            return [] if val == '[]' else val[1:-1].split(',')
    return val

def str_to_json(val: str):
    assert isinstance(val, str)
    val = fixlist.sub(r'\1"\2"\3', val)
    try:
        val = json.loads(val)
    except json.decoder.JSONDecodeError as e:
        print(val)
        raise e
    return str_to_json_recurse(val)

def tojson(thing):
    if isinstance(thing, list): return f'[{",".join(tojson(_) for _ in thing)}]'
    if hasattr(thing, 'model_dump_json'): return thing.model_dump_json()
    if hasattr(thing, 'json'): return thing.json()
    return str(thing)

def set_from_file(fname):
    if os.path.exists(fname):
        return set(Path(fname).read_text().strip().split(os.linesep))
    return set()

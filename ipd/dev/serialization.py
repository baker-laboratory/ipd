import json
import re

fixlist = re.compile(r'(^|[^"])(\[[^"]+\])($|[^"])')

def json_to_py(val):
    if isinstance(val, dict): return {k: json_to_py(v) for k, v in val.items()}
    if isinstance(val, list): return [json_to_py(v) for v in val]
    if isinstance(val, str) and val:
        if (val[0], val[-1]) == ('[', ']'):
            return [] if val == '[]' else val[1:-1].split(',')
    return val

def str_to_json(val: str):
    assert isinstance(val, str)
    val = fixlist.sub(r'\1"\2"\3', val)
    return json_to_py(json.loads(val))

def tojson(thing):
    if isinstance(thing, list): return f'[{",".join(tojson(_) for _ in thing)}]'
    if hasattr(thing, 'model_dump_json'): return thing.model_dump_json()
    if hasattr(thing, 'json'): return thing.json()
    return str(thing)

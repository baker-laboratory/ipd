from pathlib import Path
import os
import inspect
import re

import jinja2

def make_testfile(sourcefile, testifle):
    assert not os.path.exists(testifle)
    code = Path(sourcefile).read_text()
    module = {}
    exec(code, globals(), module)
    cnames = set(re.findall(r'class (.+)[\[\(:]', code))
    mnames = set(re.findall(r' def (.+)[\[\(]', code))
    fnames = set(re.findall(r'def (.+)[\[\(]', code)) - mnames
    funcs = {k: v for k, v in module.items() if inspect.isfunction(v) and k in fnames}
    funcs = {k: (v, inspect.signature(v)) for k, v in funcs.items()}
    classes = {k: v for k, v in module.items() if inspect.isclass(v) and k in cnames}
    methods = {
        n: {
            f'{n}_{k}': (v, inspect.signature(v))
            for k, v in inspect.getmembers(c) if inspect.isfunction(v) and k in mnames
        }
        for n, c in classes.items()
    }
    environment = jinja2.Environment(trim_blocks=True)
    template = environment.from_string(testfile_template)
    testcode = template.render(funcs=funcs, classes=classes, methods=methods)
    Path(testifle).write_text(testcode)
    os.system(f'yapf -i {testifle}')

testfile_template = '''
import pytest

def main():
{% for name in funcs %}
    test_{{name}}()
{% endfor %}
{% for name in classes %}
    test_{{name}}()
{% endfor %}

{% for name, (func, sig) in funcs.items() %}
def test_{{name}}():
    # {{func.__qualname__}}{{ sig }}
    assert 0

{% endfor %}

{% for clsname, cls in classes.items() %}
def test_{{clsname}}():
{% for name, (func, sig) in methods[clsname].items() %}
    # {{func.__qualname__}}{{ sig }}
{% endfor %}
    assert 0

{% endfor %}
if __name__ == '__main__':{# notmain #}

    main()

'''.lstrip()

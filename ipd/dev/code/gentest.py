from pathlib import Path
import os
import inspect
import re

import ipd

def gitpath(path: str, cwd='.') -> str:
    with ipd.dev.cd(cwd):
        module = ''
        while path and path != '/':
            path, m = os.path.split(path)
            module = f'{m}.{module}' if module else m
            if os.path.exists(os.path.join(path, '.git')):
                return module
    return ''

def qualname_of_file(sourcefile):
    if not sourcefile.endswith('.py'):
        raise ValueError(f'qualname_of_file: {sourcefile} must end with .py:')
    sourcefile = sourcefile[:-3]
    if qual := gitpath(sourcefile): return qual
    return os.path.basename(sourcefile)

def make_testfile(sourcefile, testifle):
    import jinja2
    assert not os.path.exists(testifle)
    os.makedirs(os.path.dirname(testifle), exist_ok=True)
    qualname = qualname_of_file(sourcefile)
    code = Path(sourcefile).read_text()
    prev_globals = set(globals().keys())
    exec(code, globals(), globals())
    module = {k: v for k, v in globals().items() if k not in prev_globals}
    cnames = {n for n in re.findall(r'class (.+)[\[\(:]', code) if n[0] != '_'}
    mnames = {n for n in re.findall(r' def (.+)[\[\(]', code) if n[0] != '_'}
    fnames = {n for n in re.findall(r'def (.+)[\[\(]', code) if n[0] != '_'} - mnames
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
    for n, (f, s) in funcs.items():
        f.__qualname__ = f'{qualname}.{n}'
    for c in classes:
        for n, (f, s) in methods[c].items():
            f.__qualname__ = f'{qualname}.{f.__qualname__}'
    for n, c in classes.items():
        c.__qualname__ = f'{qualname}.{n}'
    environment = jinja2.Environment(trim_blocks=True)
    template = environment.from_string(testfile_template)
    testcode = template.render(funcs=funcs, classes=classes, methods=methods, sourcefile=sourcefile)
    testcode = testcode.replace('ipd.dev.code.gentest', qualname)
    testcode = testcode.replace(f'.{qualname}', '')
    Path(testifle).write_text(testcode)
    os.system(f'yapf -i {testifle}')

testfile_template = """
import pytest

import ipd

def main():
    ipd.tests.testmain(namespace=globals())

{% for name, (func, sig) in funcs.items() %}
def test_{{name}}():
    # {{func.__name__}}{{ sig }}
    assert 0

{% endfor %}
{% for clsname, cls in classes.items() %}
def test_{{clsname}}():
{% for name, (func, sig) in methods[clsname].items() %}
    # {{clsname+'.'+func.__name__}}{{ sig }}
{% endfor %}
    assert 0

{% endfor %}
# please develop a comprehensive set of pytest tests, including edge cases and input validation, for the code in file:
# {{sourcefile}}, specifically the functions, classes and methods specified below:
{% for name, (func, sig) in funcs.items() %}
# please develop a comprehensive set of pytest tests, including edge cases and input validation, for the function {{func.__name__}} with the following signature: {{func.__qualname__}}{{ sig }}
{% endfor %}
{% for clsname, cls in classes.items() %}
# please write a comprehensive set of pytest tests, including edge cases and input validation, for the class {{clsname}} with the following member function signatures:
{% for name, (func, sig) in methods[clsname].items() %}
#        {{func.__qualname__}}{{ sig }}
{% endfor %}
{% endfor %}

if __name__ == '__main__':{# in template #}

    main()

""".lstrip()

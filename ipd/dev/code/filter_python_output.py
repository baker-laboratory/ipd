import os
import re
from ipd.bunch import Bunch

re_block = re.compile(r'  File "([^"]+)", line (\d+), in (.*)')
re_end = re.compile(r'(^[A-Za-z0-9.]+Error)(: .*)?')
re_null = r'a^'  # never matches
re_presets = dict(ipd_boilerplate=Bunch(
    file=r'ipd/tests/maintest\.py|icecream/icecream.py|/pprint.py|lazy_import.py|<.*>',
    func=
    r'<module>|main|call_with_args_from|wrapper|print_table|make_table|import_module|import_optional_dependency',
))

def filter_python_output(
    text,
    entrypoint=None,
    re_file=re_null,
    re_func=re_null,
    preset=None,
    minlines=30,
    **kw,
):
    # if entrypoint == 'codetool': return text
    if preset and re_file == re_null: re_file = re_presets[preset].file
    if preset and re_func == re_null: re_func = re_presets[preset].func
    if isinstance(re_file, str): re_file = re.compile(re_file)
    if isinstance(re_func, str): re_func = re.compile(re_func)
    result = []
    file, lineno, func, block = None, None, None, None
    skipped = []
    lines = text.splitlines()
    if len(lines) < minlines:
        return text
    for line in lines:
        if m := re_block.match(line):
            _finish_block(block, file, func, re_file, re_func, result, skipped)
            file, linene, func, block = *m.groups(), [line]
        elif m := re_end.match(line):
            _finish_block(block, file, func, re_file, re_func, result, skipped, keep=True)
            file, lineno, func, block = None, None, None, None
            result.append(_strip_extra_whitespace(line))
        elif block:
            block.append(_strip_extra_whitespace(line))
        else:
            result.append(_strip_extra_whitespace(line))
    text = os.linesep.join(result) + os.linesep
    return _filter_numpy_version_nonsense(text)

def _finish_block(block, file, func, re_file, re_func, result, skipped, keep=False):
    if block:
        filematch = re_file.search(file)
        funcmatch = re_func.search(func)
        if filematch or funcmatch and not keep:
            skipped.append(func)
        else:
            if skipped:
                # result.append('  [' + str.join('] => [', skipped) + '] =>')
                result.append('  ' + str.join('=>', skipped) + ' =>')
                skipped.clear()
            result.extend(block)

def _strip_extra_whitespace(line):
    if not line[:60].strip(): return line.strip()
    return line.rstrip()

def _filter_numpy_version_nonsense(text):
    text = text.replace(
        """
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.3 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

""", '')
    text = text.replace(
        """    from numexpr.interpreter import MAX_THREADS, use_vml, __BLOCK_SIZE1__
AttributeError: _ARRAY_API not found
""", '')
    return text

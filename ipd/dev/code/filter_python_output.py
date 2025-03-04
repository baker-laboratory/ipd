import os
import re
from ipd.bunch import Bunch

re_block = re.compile(r'  File "([^"]+)", line (\d+), in (.*)')
re_end = re.compile(r'(^[A-Z][A-Za-z0-9]+Error)(: .*)?')
re_null = r'a^'  # never matches
re_presets = dict(ipd_boilerplate=Bunch(
    file=r'ipd/tests/maintest\.py|icecream/icecream.py|/pprint.py|lazy_import.py|<.*>',
    func=r'<module>|main|call_with_args_from|wrapper|print_table|make_table|import_module',
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
        line = line.rstrip() + os.linesep
        if m := re_block.match(line):
            _finish_block(block, file, func, re_file, re_func, result, skipped)
            file, linene, func, block = *m.groups(), [line]
        elif m := re_end.match(line):
            _finish_block(block, file, func, re_file, re_func, result, skipped, keep=True)
            file, lineno, func, block = None, None, None, None
            result.append(line)
        elif block:
            block.append(line)
        else:
            result.append(line)
    return os.linesep.join(map(str.rstrip, result)) + os.linesep

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

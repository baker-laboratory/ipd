from collections import defaultdict
import os
import re
from ipd.bunch import Bunch

import ipd

re_block = re.compile(r'  File "([^"]+)", line (\d+), in (.*)')
re_end = re.compile(r'(^[A-Za-z0-9.]+Error)(: .*)?')
re_null = r'a^'  # never matches
re_presets = dict(ipd_boilerplate=Bunch(
    file=
    r'ipd/tests/maintest\.py|icecream/icecream.py|/pprint.py|lazy_import.py|<.*>|numexpr/__init__.py|hydra/_internal/defaults_list.py|click/core.py|/typer/main.py|/assertion/rewrite.py',
    func=
    r'<module>|main|call_with_args_from|wrapper|print_table|make_table|import_module|import_optional_dependency|kwcall',
))

def filter_python_output(
    text,
    entrypoint=None,
    re_file=re_null,
    re_func=re_null,
    preset=None,
    minlines=30,
    filter_numpy_version_nonsense=True,
    **kw,
):
    kw = ipd.dev.project_local_config('filter_python_output') | kw
    assert kw['keep_blank_lines']
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
        line = _strip_line_extra_whitespace(line)
        if not line.strip() and not kw['keep_blank_lines']: continue
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
    text = os.linesep.join(result) + os.linesep
    if filter_numpy_version_nonsense:
        text = _filter_numpy_version_nonsense(text)
    return text

def _finish_block(block, file, func, re_file, re_func, result, skipped, keep=False):
    if block:
        filematch = re_file.search(file)
        funcmatch = re_func.search(func)
        if filematch or funcmatch and not keep:
            file = os.path.basename(file.replace('/__init__.py', '[init]'))
            skipped.append(file if func == '<module>' else func)
        else:
            if skipped:
                # result.append('  [' + str.join('] => [', skipped) + '] =>')
                result.append('  ' + str.join(' -> ', skipped) + ' ->')
                skipped.clear()
            result.extend(block)

def _strip_line_extra_whitespace(line):
    if not line[:60].strip(): return line.strip()
    return line.rstrip()

# def _strip_text_extra_whitespace(text):
# return re.sub(r'\n\n', os.linesep, text, re.MULTILINE)

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
        """A module that was compiled using NumPy 1.x cannot be run in
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
    text = text.replace("""AttributeError: _ARRAY_API not found



Traceback""", '')
    return text

'''Traceback (most recent call last):
  File "example.py", line 10, in <module>
    1/0
ZeroDivisionError: division by zero
foof
ISNR'''

def analyze_python_errors_log(text):
    # traceback_pattern = re.compile(r'Traceback \(most recent call last\):.*?\n[A-Za-z]+?Error:.*?$', re.DOTALL)
    traceback_pattern = re.compile(r'Traceback \(most recent call last\):.*?(?=\nTraceback |\Z)', re.DOTALL)
    file_line_pattern = re.compile(r'\n\s*File "(.*?\.py)", line (\d+), in ')
    error_pattern = re.compile(r'\n\s*[A-Za-z_0-9]+Error: .*')
    """Analyze Python error logs and create a report of unique stack traces.

    Args:
        text (str): The log file content as a string.

    Returns:
        str: A report of unique stack traces.

    Example:
        >>> log = '''Traceback (most recent call last):
        ...   File "example.py", line 10, in <module>
        ...     1/0
        ... ZeroDivisionError: division by zero'''
        >>> result = analyze_python_errors_log(log)
        >>> 'Unique Stack Traces Report (1 unique traces):' in result
        True
    """
    trace_map = defaultdict(list)
    tracebacks = traceback_pattern.findall(text)
    for trace in tracebacks:
        filematch = file_line_pattern.search(trace)
        errmatch = error_pattern.search(trace)
        assert filematch and errmatch, f'Error pattern not found in {trace}'
        location = ':'.join(filematch.groups())
        error = errmatch.group(0).strip()
        key = (location, error)
        if key not in trace_map:
            trace_map[key] = trace
    return create_errors_log_report(trace_map)

def create_errors_log_report(trace_map):
    """Generate a report from a map of unique stack traces.

    Args:
        trace_map (dict): A dictionary where keys are unique error signatures
            and values are corresponding stack traces.

    Returns:
        str: A formatted report of the unique stack traces.

    Example:
        >>> trace_map = {('1/0', 'division by zero'): '''Traceback (most recent call last):
        ...   File "example.py", line 10, in <module>
        ...     1/0
        ... ZeroDivisionError: division by zero'''}
        >>> report = create_errors_log_report(trace_map)
        >>> 'Unique Stack Traces Report (1 unique traces):' in report
        True
    """
    with ipd.capture_stdio() as printed:
        print(f"Unique Stack Traces Report ({len(trace_map)} unique traces):")
        print("="*80 + "\n")
        for (_, trace) in trace_map.items():
            print(trace)
            print("-"*80 + "\n")
    return printed.read()

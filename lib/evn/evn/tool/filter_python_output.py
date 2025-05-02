import os
import re

def main():
    print('filter_python_output main')
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfiles', type=str, nargs='+')
    parser.add_argument('-m', '--minlines', type=int, default=0)
    parser.add_argument('-p', '--preset', type=str, default='boilerplate')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(sys.argv[1:]).__dict__
    for fname in args['inputfiles']:
        os.rename(fname, f'{fname}.orig')
        with open(f'{fname}.orig', 'r') as inp:
            text = inp.read()
        newtext = filter_python_output(text, **args)
        with open(fname, 'w') as out:
            out.write(newtext)
        print(f'file: {fname}, lines {len(text.splitlines())} -> {len(newtext.splitlines())}')

presets = dict(
    unittest=dict(
        refile=
        (r'quicktest\.py|icecream/icecream.py|/pprint.py|lazy_import.py|<.*>|numexpr/__init__.py|hydra/_internal/defaults_list.py|click/core.py|/typer/main.py|/assertion/rewrite.py'
         ),
        refunc=
        (r'<module>|main|call_with_args_from|wrapper|print_table|make_table|import_module|import_optional_dependency|kwcall'
         ),
        minlines=30,
    ),
    boilerplate=dict(
        refile=
        (r'quicktest\.py|icecream/icecream.py|/pprint.py|lazy_import.py|<.*>|numexpr/__init__.py|hydra/_internal/defaults_list.py|click/core.py|/typer/main.py|/assertion/rewrite.py|/_[A-Za-z0-9i]*.py|site-packages/_pytest/.*py|evn/dev/inspect.py'
         ),
        refunc=
        (r'|main|call_with_args_from|wrapper|print_table|make_table|import_module|import_optional_dependency|kwcall'
         ),
        minlines=30,
    ),
    aggressive=dict(
        refile=
        (r'quicktest\.py|icecream/icecream.py|/pprint.py|lazy_import.py|<.*>|numexpr/__init__.py|hydra/_internal/defaults_list.py|click/core.py|/typer/main.py|/assertion/rewrite.py|/_[A-Za-z0-9i]*.py|site-packages/_pytest/.*py|<module>|evn/contexts.py|multipledispatch/dispatcher.py|evn/dev/inspect.py|meta/kwcall.py'
         ),
        refunc=
        (r'<module>|main|call_with_args_from|wrapper|print_table|make_table|import_module|import_optional_dependency|kwcall|main|kwcall'
         ),
        minlines=1,
    ),
)

re_blank = re.compile(r'(?:^[ \t]*\n){2,}', re.MULTILINE)
re_block = re.compile(r'  File "([^"]+)", line (\d+), in (.*)')
re_end = re.compile(r'(^[A-Za-z0-9.]+Error)(: .*)?')
re_null = r'a^'  # never matches

def filter_python_output(
    text,
    entrypoint=None,
    re_file=re_null,
    re_func=re_null,
    preset='boilerplate',
    minlines=-1,
    filter_numpy_version_nonsense=True,
    keep_blank_lines=False,
    arrows=True,
    **kw,
):
    preset = presets[preset]
    # if entrypoint == 'codetool': return text
    if minlines < 0: minlines = preset['minlines']
    if preset and re_file == re_null: re_file = preset['refile']
    if preset and re_func == re_null: re_func = preset['refunc']
    if isinstance(re_file, str): re_file = re.compile(re_file)
    if isinstance(re_func, str): re_func = re.compile(re_func)
    if text.count(os.linesep) < minlines: return text
    if filter_numpy_version_nonsense:
        text = _filter_numpy_version_nonsense(text)
    if not keep_blank_lines:
        text = re_blank.sub(os.linesep * 2, text)

    skipped = []
    result = []
    file, _lineno, func, block = None, None, None, None
    for line in text.splitlines():
        line = strip_line_extra_whitespace(line)
        if m := re_block.match(line):
            _finish_block(preset, arrows, block, file, func, re_file, re_func, result, skipped)
            file, _linene, func, block = *m.groups(), [line]
        elif m := re_end.match(line):
            _finish_block(preset, arrows, block, file, func, re_file, re_func, result, skipped, keep=True)
            file, _lineno, func, block = None, None, None, None
            result.append(line)
        elif block:
            block.append(line)
        else:
            if m := re_file_alt.match(line):
                line = transform_fileref_to_python_format(line, m)
            result.append(line)
    if result[-1]: result.append('')
    new = os.linesep.join(result)
    return new

re_file_alt = re.compile(r'^E?\s*(.+?\.py):([0-9]+): .*')
def transform_fileref_to_python_format(line, match=None):
    """
    examples
    '  File "/home/sheffler/evn/evn/tests/_prelude/test_chrono.py", line 96, ...'
    /home/sheffler/evn/evn/cli/__init__.py:32: DocTestFailure
    """
    match = match or re_file_alt.match(line)
    return f'  File "{match.group(1)}", line {match.group(2)}, ...'

def _finish_block(preset, arrows, block, file, func, re_file, re_func, result, skipped, keep=False):
    if block:
        filematch = re_file.search(file)
        funcmatch = re_func.search(func)
        if filematch or funcmatch and not keep:
            file = os.path.basename(file.replace('/__init__.py', '[init]'))
            skipped.append(file if func == '<module>' else func)
        else:
            if skipped and arrows:
                # result.append('  [' + str.join('] => [', skipped) + '] =>')
                result.append('  ' + str.join(' -> ', skipped) + ' ->')
                skipped.clear()
            result.extend(block)

def strip_line_extra_whitespace(line):
    if not line[:60].strip():
        return line.strip()
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

""",
        '',
    )
    text = text.replace(
        """A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.3 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.
""",
        '',
    )
    text = text.replace(
        """    from numexpr.interpreter import MAX_THREADS, use_vml, __BLOCK_SIZE1__
AttributeError: _ARRAY_API not found
""",
        '',
    )
    text = text.replace(
        """AttributeError: _ARRAY_API not found



Traceback""",
        '',
    )
    return text

"""Traceback (most recent call last):
  File "example.py", line 10, in <module>
    1/0
ZeroDivisionError: division by zero
foof
ISNR"""

def analyze_python_errors_log(text):
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
    # traceback_pattern = re.compile(r'Traceback \(most recent call last\):.*?\n[A-Za-z]+?Error:.*?$', re.DOTALL)
    from collections import defaultdict
    traceback_pattern = re.compile(r'Traceback \(most recent call last\):.*?(?=\nTraceback |\Z)', re.DOTALL)
    file_line_pattern = re.compile(r'\n\s*File "(.*?\.py)", line (\d+), in ')
    error_pattern = re.compile(r'\n\s*[A-Za-z_0-9]+Error: .*')
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
        >>> trace_map = {
        ...     ('1/0', 'division by zero'): '''Traceback (most recent call last):
        ...   File "example.py", line 10, in <module>
        ...     1/0
        ... ZeroDivisionError: division by zero'''
        ... }
        >>> report = create_errors_log_report(trace_map)
        >>> 'Unique Stack Traces Report (1 unique traces):' in report
        True
    """
    import evn
    with evn.capture_stdio() as printed:
        print(f'Unique Stack Traces Report ({len(trace_map)} unique traces):')
        print('='*80 + '\n')
        for _, trace in trace_map.items():
            print(trace)
            print('-'*80 + '\n')
    return printed.read()

if __name__ == '__main__':  # ignore
    main()

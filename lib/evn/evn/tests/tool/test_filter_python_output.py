import re
import difflib
import pytest
import evn

def main():
    evn.testing.quicktest(globals(), verbose=True)

def helper_test_filter_python_output(text, ref, preset):
    result = evn.tool.filter_python_output(text, preset=preset, minlines=0)
    if result != ref:
        diff = difflib.ndiff(result.splitlines(), ref.splitlines())
        # print('\nDIFF:', flush=True)
        # print('\n'.join(f'NDIFF {d}' for d in diff), flush=True)
        print('--------------------------------------------------------------------')
        print(result)
        print('--------------------------------------------------------------------')
        assert len(result.splitlines()) == len(ref.splitlines())
        assert 0, 'filter mismatch'


def test_transform_fileref_to_python_format():
    from evn.tool.filter_python_output import transform_fileref_to_python_format as tf
    new = tf('evn/_prelude/chrono.py:63: TypeError')
    assert new == '  File "evn/_prelude/chrono.py", line 63, ...'
    new2 = tf('/home/sheffler/evn/evn/cli/__init__.py:32: DocTestFailure')
    assert new2 == '  File "/home/sheffler/evn/evn/cli/__init__.py", line 32, ...'

# @pytest.mark.xfail
def test_filter_python_output_whitespace():
    result = evn.tool.filter_python_output('    \n' * 22, preset='unittest', minlines=0)
    print(result.count('\n'))
    assert result.count('\n') == 1

def test_filter_python_output_mid():
    helper_test_filter_python_output(midtext, midfiltered, preset='unittest')

def test_filter_python_output_small():
    helper_test_filter_python_output(smalltext, smallfiltered, preset='unittest')

def test_filter_python_output_error():
    helper_test_filter_python_output(errortext, errorfiltered, preset='unittest')

def test_analyze_python_errors_log():
    log = """Traceback (most recent call last):
  File "example.py", line 10, in <module>
    1/0
ZeroDivisionError: division by zero"""
    result = evn.tool.analyze_python_errors_log(log)
    # print(result)
    assert 'Unique Stack Traces Report (1 unique traces):' in result
    assert 'ZeroDivisionError: division by zero' in result

def test_create_errors_log_report():
    trace_map = {
        ('1/0', 'division by zero'):
        """Traceback (most recent call last):
  File "example.py", line 10, in <module>
    1/0
ZeroDivisionError: division by zero"""
    }

    report = evn.tool.create_errors_log_report(trace_map)
    assert 'Unique Stack Traces Report (1 unique traces):' in report
    assert 'ZeroDivisionError: division by zero' in report

def test_multiple_unique_traces():
    log = """Traceback (most recent call last):
  File "example.py", line 10, in <module>
    1/0
ZeroDivisionError: division by zero

Traceback (most recent call last):
  File "example.py", line 20, in <module>
    x = int("abc")
ValueError: invalid literal for int()"""

    result = evn.tool.analyze_python_errors_log(log)
    assert 'Unique Stack Traces Report (2 unique traces):' in result
    assert 'ZeroDivisionError: division by zero' in result
    assert 'ValueError: invalid literal for int()' in result

def test_similar_traces_are_grouped():
    log = """Traceback (most recent call last):
  File "example.py", line 13, in <module>
    1/0
ZeroDivisionError: division by zero

Traceback (most recent call last):
  File "example.py", line 13, in <module>
    1/0
ZeroDivisionError: division by zero"""

    result = evn.tool.analyze_python_errors_log(log)
    assert 'Unique Stack Traces Report (1 unique traces):' in result
    assert 'ZeroDivisionError: division by zero' in result
    assert result.count('ZeroDivisionError') == 1

def test_different_lines_are_not_grouped():
    log = """Traceback (most recent call last):
  File "example.py", line 10, in <module>
    1/0
ZeroDivisionError: division by zero

Traceback (most recent call last):
  File "example.py", line 15, in <module>
    1/0
ZeroDivisionError: division by zero"""

    result = evn.tool.analyze_python_errors_log(log)
    assert 'Unique Stack Traces Report (2 unique traces):' in result
    assert 'ZeroDivisionError: division by zero' in result
    assert result.count('ZeroDivisionError') == 2

# ######################### test data #######################
errortext = """quicktest /home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py:
Traceback (most recent call last):
  File "/home/sheffler/rfd/TEST/tests/quicktest.py", line 40, in quicktest
    _quicktest_run_test_function(name, func, result, nofail, fixtures, funcsetup, kw)
  File "/home/sheffler/rfd/TEST/tests/quicktest.py", line 80, in _quicktest_run_test_function
    TEST.dev.call_with_args_from(fixtures, func, **kw)
  File "/home/sheffler/rfd/TEST.dev.decorators.py", line 33, in call_with_args_from
    return func(**args)
           ^^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 19, in test_filter_python_output_small
    helper_test_filter_python_output(smalltext, smallfiltered, preset='boilerplate')
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'helper_test_filter_python_output' is not defined
Times(name=Timer, order=longest, summary=sum):
    quicktest *    0.39438
============== run_tests_for_file.py done, time   0.611 ==============
"""
errorfiltered = """quicktest /home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py:
Traceback (most recent call last):
  quicktest -> _quicktest_run_test_function -> call_with_args_from ->
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 19, in test_filter_python_output_small
    helper_test_filter_python_output(smalltext, smallfiltered, preset='boilerplate')
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'helper_test_filter_python_output' is not defined
Times(name=Timer, order=longest, summary=sum):
    quicktest *    0.39438
============== run_tests_for_file.py done, time   0.611 ==============
"""

midtext = """quicktest /home/sheffler/rfd/lib/TEST/TEST/tests/sym/test_sym_detect.py:
==============test_sym_detect_frames_noised_T===============
Traceback (most recent call last):
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/sym/test_sym_detect.py", line 185, in <module>
    main()
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/sym/test_sym_detect.py", line 25, in main
    TEST.tests.quicktest(namespace=globals(), config=config_test, verbose=1)
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/quicktest.py", line 40, in quicktest
    _quicktest_run_test_function(name, func, result, nofail, fixtures, funcsetup, kw)
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/quicktest.py", line 93, in _quicktest_run_test_function
    elif error: raise error
                ^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/quicktest.py", line 80, in _quicktest_run_test_function
    TEST.dev.call_with_args_from(fixtures, func, **kw)
  File "/home/sheffler/rfd/lib/TEST/TEST.dev.decorators.py", line 33, in call_with_args_from
    return func(**args)
           ^^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/sym/test_sym_detect.py", line 76, in func_noised
    sinfo = helper_test_frames(nframes, symid, ideal=False)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/sym/test_sym_detect.py", line 30, in helper_test_frames
    TEST.icv(tol)
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/site-packages/icecream/icecream.py", line 208, in __call__
    out = self._format(callFrame, *args)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/site-packages/icecream/icecream.py", line 242, in _format
    out = self._formatArgs(
          ^^^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/site-packages/icecream/icecream.py", line 255, in _formatArgs
    out = self._constructArgumentOutput(prefix, context, pairs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/site-packages/icecream/icecream.py", line 262, in _constructArgumentOutput
    pairs = [(arg, self.argToStringFunction(val)) for arg, val in pairs]
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/functools.py", line 909, in wrapper
    return dispatch(args[0].__class__)(*args, **kw)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/site-packages/icecream/icecream.py", line 183, in argumentToString
    s = DEFAULT_ARG_TO_STRING_FUNCTION(obj)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/pprint.py", line 62, in pformat
    underscore_numbers=underscore_numbers).pformat(object)
                                           ^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/pprint.py", line 161, in pformat
    self._format(object, sio, 0, 0, {}, 0)
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/pprint.py", line 178, in _format
    rep = self._repr(object, context, level)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/pprint.py", line 458, in _repr
    repr, readable, recursive = self.format(object, context.copy(),
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/pprint.py", line 471, in format
    return self._safe_repr(object, context, maxlevels, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/sw/MambaForge/envs/rfdsym312/lib/python3.12/pprint.py", line 632, in _safe_repr
    rep = repr(object)
          ^^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/dev/tolerances.py", line 52, in __repr__
    TEST.dev.print_table(self.kw)
  File "/home/sheffler/rfd/lib/TEST/TEST/dev/format.py", line 21, in print_table
    table = make_table(thing, **kw)
            ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/dev/format.py", line 14, in make_table
    if isinstance(thing, dict): return make_table_dict(thing, **kw)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/dev/format.py", line 37, in make_table_dict
    assert isinstance(mapping, Mapping) and mapping
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Times(name=Timer, order=longest, summary=sum):
    test_sym_detect.py:func_noised      0.43416
                          quicktest *    0.39847
                     sym.py:frames      0.03893
Traceback (most recent call last):
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 92, in <module>
    main()
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 6, in main
    TEST.tests.quicktest(namespace=globals())
  File "/home/sheffler/rfd/TEST/tests/quicktest.py", line 40, in quicktest
    _quicktest_run_test_function(name, func, result, nofail, fixtures, funcsetup, kw)
  File "/home/sheffler/rfd/TEST/tests/quicktest.py", line 93, in _quicktest_run_test_function
    elif error: raise error
                ^^^^^^^^^^^
  File "/home/sheffler/rfd/TEST/tests/quicktest.py", line 80, in _quicktest_run_test_function
    TEST.dev.call_with_args_from(fixtures, func, **kw)
  File "/home/sheffler/rfd/TEST.dev.decorators.py", line 33, in call_with_args_from
    return func(**args)
           ^^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 13, in test_filter_python_output
    assert 0
           ^
AssertionError

"""

midfiltered = """quicktest /home/sheffler/rfd/lib/TEST/TEST/tests/sym/test_sym_detect.py:
==============test_sym_detect_frames_noised_T===============
Traceback (most recent call last):
  test_sym_detect.py -> main -> quicktest -> _quicktest_run_test_function -> _quicktest_run_test_function -> call_with_args_from ->
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/sym/test_sym_detect.py", line 76, in func_noised
    sinfo = helper_test_frames(nframes, symid, ideal=False)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/sym/test_sym_detect.py", line 30, in helper_test_frames
    TEST.icv(tol)
  __call__ -> _format -> _formatArgs -> _constructArgumentOutput -> wrapper -> argumentToString -> pformat -> pformat -> _format -> _repr -> format -> _safe_repr ->
  File "/home/sheffler/rfd/lib/TEST/TEST/dev/tolerances.py", line 52, in __repr__
    TEST.dev.print_table(self.kw)
  print_table -> make_table ->
  File "/home/sheffler/rfd/lib/TEST/TEST/dev/format.py", line 37, in make_table_dict
    assert isinstance(mapping, Mapping) and mapping
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Times(name=Timer, order=longest, summary=sum):
    test_sym_detect.py:func_noised      0.43416
                          quicktest *    0.39847
                     sym.py:frames      0.03893
Traceback (most recent call last):
  test_filter_python_output.py -> main -> quicktest -> _quicktest_run_test_function -> _quicktest_run_test_function -> call_with_args_from ->
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 13, in test_filter_python_output
    assert 0
           ^
AssertionError
"""

smalltext = """extra text at the start
Traceback (most recent call last):
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 92, in <module>
    main()
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 6, in main
    TEST.tests.quicktest(namespace=globals())
  File "/home/sheffler/rfd/TEST/tests/quicktest.py", line 40, in quicktest
    _quicktest_run_test_function(name, func, result, nofail, fixtures, funcsetup, kw)
  File "/home/sheffler/rfd/TEST/tests/quicktest.py", line 93, in _quicktest_run_test_function
    elif error: raise error
                ^^^^^^^^^^^
  File "/home/sheffler/rfd/TEST/tests/quicktest.py", line 80, in _quicktest_run_test_function
    TEST.dev.call_with_args_from(fixtures, func, **kw)
  File "/home/sheffler/rfd/TEST.dev.decorators.py", line 33, in call_with_args_from
    return func(**args)
           ^^^^^^^^^^^^
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 13, in test_filter_python_output
    assert 0
           ^
AssertionError
extra text at the end
"""

smallfiltered = """extra text at the start
Traceback (most recent call last):
  test_filter_python_output.py -> main -> quicktest -> _quicktest_run_test_function -> _quicktest_run_test_function -> call_with_args_from ->
  File "/home/sheffler/rfd/lib/TEST/TEST/tests/dev/code/test_filter_python_output.py", line 13, in test_filter_python_output
    assert 0
           ^
AssertionError
extra text at the end
"""

if __name__ == '__main__':
    main()

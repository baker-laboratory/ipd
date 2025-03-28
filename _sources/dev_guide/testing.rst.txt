Testing
==========

All of your nontrivial functions and class methods shold have tests. All modules ipd/foo/bar/baz.py should have a corresponding test file ipd/tests/foo/bar/test_baz.py. All of the nice tooling available assumes this structure.

Tools for running a particular test file (in your editor)
-----------------------------------------------------------
IPD includs lots of nice tooling to make writing and running tests easy, particularly if you integrate them into your ide. For example, :py:func:`ipd.dev.make_testfile` will create a test file stencil for you.

ipd test file <package> <module>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``ipd test file <package> <path to module or test`` will run the tests for a module (or other test actions, like validate pyproject.toml) if a testfile is available, or else run the module if it has a main function. You can also run ``ipd/tools/run_tests_on_file.py`` is that is easier. Highly recommend adding this to your editor with a hotkey to run the current file. Examples:

.. code-block:: bash

    ipd test file ipd ipd/sym/sym.py
    # runs ipd/tests/sym/test_sym.py
    ipd test file ipd ipd/new/module.py
    # creates stencil and opens ipd/tests/new/test_module.py
    ipd test file ipd pyproject.toml
    # runs validate-pyproject on pyproject.toml
    ipd test file ipd noxfile.py
    # runs nox
    ipd test file foo foo/myapp.py
    # runs foo/myapp.py (if it has a main function)
    # runs foo/tests/test_myapp.py
    #   (if myapp has no main function and foo/tests/test_myapp.py has a main)ipd test file foo foo/myapp.py
    # runs pytest foo/tests/test_myapp.py
    #   (if myapp has no main function and foo/tests/test_myapp.py has no main function)

ipd.tests.maintest
~~~~~~~~~~~~~~~~~~~~
Fast test "runner" that will run tests in a namespace kinda as pytest would do, but with less overhead and output more suitable for an ide. Invoked for a modules tests with ``ipd.tests.maintest(globals())``. Auto generated test stencils will use this runner.

Swapping between module file and test file
---------------------------------------------
Highly recomment adding a hotkey to your editor to swap between a module file and its test file. Here is an example for sublime text:

.. code-block:: python

    def get_project_root(path: str) -> str:
        # Search upward for pyproject.toml to detect project root
        current = os.path.abspath(path)
        while current != os.path.dirname(current):
            if os.path.exists(os.path.join(current, 'pyproject.toml')):
                return current
            current = os.path.dirname(current)
        return '/'

    class SwapSourceTestCommand(sublime_plugin.TextCommand):
        def run(self, edit):
            current_file = self.view.file_name()
            if not current_file: return
            project_root = get_project_root(current_file)
            if not project_root: return
            relative_path = os.path.relpath(current_file, project_root)
            parts = relative_path.split(os.sep)
            if "tests" in parts:
                parts.remove("tests")
                parts[-1] = parts[-1].replace("test_", "")
            else:
                parts.insert(1, "tests")
                parts[-1] = f"test_{parts[-1]}"
            related_path = os.path.join(*parts)
            related_file = os.path.join(project_root, related_path)
            if os.path.exists(related_file): self.view.window().open_file(related_file)



Running all Tests
-------------------

pytest
~~~~~~~~~
run and/or doctest everything in your current python environment

See settings in `pyproject.toml`.

To run in your local environment:
    ``pytest -n4 --doctest-modules``

To run in general:
    ``uv run --extras all pytest -n4 --doctest-modules``

nox
~~~~~
run and doctest everything in a test matrix (uses a fair bit of disk space)

To run the full test matrix, run:
    ``uv tool run nox``

To run just one config:
    ``nox -s 3.9``
    ``nox -- 3.12 all``

To run singlne threaded:
    ``nox -- 1 3.12 all``

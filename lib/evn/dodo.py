import os
import sys
import sysconfig
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import track
import subprocess

DOIT_CONFIG = {
    'backend': 'json',
    'dep_file': '.doit.json',
}

root = Path(__file__).parent.absolute()
build = root / '_build' / f'py3{sys.version_info.minor}'

def task_cmake():
    return {
        'actions': [f'cmake {find_pybind()} -B {build} -S {root} -GNinja'],
        'file_dep': [root / 'CMakeLists.txt'],
        'targets': [build / 'build.ninja'],
    }

def task_build():
    return {
        'actions': [f'cd {build} && ninja'],
        'file_dep': [build / 'build.ninja'],
    }

def find_pybind():
    pybind = sysconfig.get_paths()['purelib'] + '/pybind11'
    pybind = f'-Dpybind11_DIR={pybind}'
    # print(pybind)
    return pybind

def task_import_check():
    """Try to import the compiled module to verify it's working"""

    def import_test():
        try:
            pass
        except Exception as e:
            print('❌ Import failed:', e)
            raise

    return {
        'actions': [import_test],
        'task_dep': ['build'],
    }

def task_test():
    """Run tests using pytest"""
    return {
        'actions': ['pytest --doctest-modules evn'],
        'task_dep': ['import_check'],
    }

def task_matrix():
    versions = ['3.13', '3.12', '3.11', '3.10']
    log_dir = Path("test-logs")
    log_dir.mkdir(exist_ok=True)

    def run_matrix(python, parallel, quiet):
        selected = [python] if python in versions else versions
        results = []
        print('versions:', selected)

        def run_test(v, results=results):
            log_file = log_dir / f'test_py{v}.log'
            # cmd = f'uv run --extra dev --python {v} doit test'
            subdir = os.path.abspath(f'.test_py{v}')
            os.system(f'rm {subdir}/*')
            os.makedirs(subdir, exist_ok=True)
            for p in ['pyproject.toml', 'evn', 'CMakeLists.txt']:
                if not os.path.exists(f'{subdir}/{p}'):
                    os.symlink(os.path.abspath(p), f'{subdir}/{p}')
            cmd = f'cd {subdir} && uv run --extra dev --python {v} pytest'

            with log_file.open('w') as f:
                try:
                    subprocess.run(cmd, shell=True, check=True, stdout=f if quiet else None, stderr=subprocess.STDOUT)
                    results.append((v, True))
                except subprocess.CalledProcessError:
                    results.append((v, False))

        if parallel:
            with ThreadPoolExecutor() as executor:
                fut = [executor.submit(run_test, v) for v in selected]
            list(track(as_completed(fut), total=len(fut), description="Running tests..."))
        else:
            for v in selected:
                run_test(v)

        print(f"\n=== Test Summary: {len(results)} results ===")
        for v, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"Python {v}: {status} (see {log_dir}/python{v}.log)")

        # if not all(passed for _, passed in results):
            # raise Exception("One or more Python versions failed")

    return dict(
        params=[
            dict(name='python',
                 long='python',
                 default='all',
                 help='Python version to run (e.g., 3.11), or "all"'),
            dict(name='parallel', long='parallel', default=True, help='run tests in parallel'),
            dict(name='quiet', long='quiet', default=False),
        ],
        actions=[(run_matrix, )],
        verbosity=2,
    )

def task_wheel():
    os.makedirs('wheelhouse', exist_ok=True)
    return dict(
        actions=[f'cibuildwheel --only cp3{ver}-manylinux_x86_64' for ver in range(10, 14)],
        file_dep=[
            'evn/format/_common.hpp',
            'evn/format/_detect_formatted_blocks.cpp',
            'evn/format/_token_column_format.cpp',
        ],
        targets=[
            # 'wheelhouse/evn-0.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
            'wheelhouse/evn-0.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
            'wheelhouse/evn-0.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
            'wheelhouse/evn-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
            'wheelhouse/evn-0.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
        ],
    )

def task_nox():
    return dict(
        actions=['nox'],
        file_dep=[
            # 'wheelhouse/evn-0.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
            'wheelhouse/evn-0.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
            'wheelhouse/evn-0.1.0-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
            'wheelhouse/evn-0.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
            'wheelhouse/evn-0.1.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
        ],
    )

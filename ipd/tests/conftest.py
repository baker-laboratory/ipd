import time
import os
import pytest
from ipd.tests import fixtures

@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(os.path.dirname(__file__) + '/..')

@pytest.fixture
def pdbfname():
    return fixtures.pdbfname()

@pytest.fixture
def pdbfnames():
    return fixtures.pdbfnames()

@pytest.fixture
def pdbcontents():
    return fixtures.pdbcontents()

@pytest.fixture
def three_PDBFiles():
    return fixtures.three_PDBFiles()

@pytest.fixture
def pdbfile():
    return fixtures.pdbfile()

@pytest.fixture
def ncac():
    return fixtures.ncac()

@pytest.fixture
def ncaco():
    return fixtures.ncaco()

@pytest.fixture
def pdb1pgx():
    return fixtures.pdb1pgx()

@pytest.fixture
def pdb1coi():
    return fixtures.pdb1coi()

@pytest.fixture
def pdb1qys():
    return fixtures.pdb1qys()
import pytest

timings = {}
_start_times = {}

def mark_start(label):
    _start_times[label] = time.perf_counter()

def mark_end(label):
    timings[label] = time.perf_counter() - _start_times.get(label, 0)

def pytest_configure(config):
    if config.pluginmanager.hasplugin("xdist") and hasattr(config, "workerinput"):
        config._worker_start_time = time.perf_counter()
    mark_start("session")

def pytest_sessionstart(session):
    if hasattr(session.config, "_worker_start_time"):
        timings["worker_startup"] = time.perf_counter() - session.config._worker_start_time
    mark_start("collection")

def pytest_collection_modifyitems(session, config, items):
    mark_end("collection")
    mark_start("runtestloop")

def pytest_sessionfinish(session, exitstatus):
    mark_end("runtestloop")
    mark_end("session")
    if hasattr(session.config, "workerinput"):
        return
    print("\n🔍 Pytest Timings:")
    for k, v in sorted(timings.items()):
        print(f"  {k:<16}: {v:.2f} seconds")

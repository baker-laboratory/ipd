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

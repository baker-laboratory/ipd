import pytest

pytest.importorskip('pymol')
import contextlib
import tempfile

from fastapi.testclient import TestClient

import ipd.ppp

@contextlib.contextmanager
def ppp_test_stuff():
    with tempfile.TemporaryDirectory() as tmpdir:
        server, backend, client = ipd.ppp.server.run(
            port=12346,
            dburl=f'sqlite:////{tmpdir}/test.db',
            workers=1,
            loglevel='warning',
        )
        ipd.ppp.server.defaults.ensure_init_db(backend)
        testclient = TestClient(backend.app)
        try:
            yield backend, server, client, testclient
        finally:
            server.stop()

@pytest.fixture(scope='module')
def ppp():
    with ppp_test_stuff() as stuff:
        yield stuff

@pytest.fixture(scope='function')
def ppp_per_func(ppp):
    ppp[0]._clear_all_data_for_testing_only()
    ipd.ppp.server.defaults.add_defaults()
    return ppp

@pytest.fixture(scope='function')
def pppbackend(ppp_per_func):
    return ppp_per_func[0]

@pytest.fixture(scope='function')
def pppserver(ppp_per_func):
    return ppp_per_func[1]

@pytest.fixture(scope='function')
def pppclient(ppp_per_func):
    return ppp_per_func[2]

@pytest.fixture(scope='function')
def ppptestclient(ppp_per_func):
    return ppp_per_func[3]

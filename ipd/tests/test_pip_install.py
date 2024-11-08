import pytest

import ipd

@pytest.mark.ci
def test_pip_install(tmpdir):
    try:
        result = ipd.dev.run(f'python -mvenv {tmpdir} && time {tmpdir}/bin/pip install {ipd.projdir}/..')
    except RuntimeError as e:
        print(e)
        assert False, str(e)

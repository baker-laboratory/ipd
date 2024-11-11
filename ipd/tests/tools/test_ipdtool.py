import os
import pytest

typer = pytest.importorskip('typer')
pytest.mark.skipif(int(typer.__version__.split('.')[1]) < 12, reason='ipd.sym.Helix breaks on numpy 2')

from typer.testing import CliRunner
import ipd

runner = CliRunner()
ipdtool = ipd.tools.IPDTool()

def run(cmd):
    print('run:', cmd)
    if isinstance(cmd, str): cmd = cmd.split()
    result = runner.invoke(ipdtool.__app__, cmd)
    if result.exit_code:
        print(result.stdout)
    assert not result.exit_code
    return result

def main():
    test_ipdtool_basic()
    # test_ipdtool_ci_update()
    test_setup_submodules()

@pytest.mark.fast
def test_ipdtool_basic():
    result = run('')
    assert "Usage" in result.stdout
    result = run('ci')
    assert "Usage" in result.stdout
    result = run('ci repo')
    assert "Usage" in result.stdout

@pytest.mark.skip
def test_setup_submodules():
    result = run('ci repo setup_submodules --path /tmp/rf_diffusion')

# @pytest.mark.skipif(not os.path.isdir(os.path.expanduser('~/bare_repos')), reason='no bare_repos dir')
@pytest.mark.skip
def test_ipdtool_ci_update():
    if not os.path.isdir(os.path.expanduser('~/bare_repos')): return
    result = run('ci update_repos ~/bare_repos')
    print(result.stdout)

if __name__ == '__main__':
    main()

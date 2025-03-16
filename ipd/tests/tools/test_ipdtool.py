import os
import pytest
import re

typer = pytest.importorskip('typer')
pytest.mark.skipif(int(typer.__version__.split('.')[1]) < 12, reason='ipd.sym.Helix breaks on numpy 2')

from typer.testing import CliRunner
import ipd

def main():
    ipd.tests.maintest(globals())

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

def normalize_text(text):
    """Normalize text by removing whitespace and newlines to make comparisons more robust."""
    # Remove all whitespace (spaces, tabs, newlines)
    return re.sub(r'\s+', '', text)

def test_ipdtool_basic():
    # Test the basic commands, normalizing output to handle formatting variations
    result = run('')
    normalized_output = normalize_text(result.stdout)
    assert "Usage" in normalized_output, f"Original output: {result.stdout}"

    result = run('ci')
    normalized_output = normalize_text(result.stdout)
    assert "Usage" in normalized_output, f"Original output: {result.stdout}"

    result = run('ci repo')
    normalized_output = normalize_text(result.stdout)
    assert "Usage" in normalized_output, f"Original output: {result.stdout}"

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

import os

import pytest
from rich import print
from typer.testing import CliRunner

import ipd

def main():
    test_citool_update_library()
    # test_clitool_pytest()
    # test_clitool_pytest_slurm_parallel()
    print('test clitool PASS')

runner = CliRunner()

@pytest.mark.ci
def test_citool_update_library():
    tool = ipd.tools.IPDTool()
    result = runner.invoke(tool.__app__, 'ci update_library')
    print(result.stdout)
    assert result.exit_code == 0

@pytest.mark.xfail
def test_that_fails():
    assert 0

def test_foo():
    pass

def test_bar():
    pass

os.chdir(f'{os.path.dirname(__file__)}/../../../..')

# @pytest.mark.recursive
# def test_clitool_pytest():
#     tool = ipd.tools.ci.TestsTool()
#     for cmd, result, log in tool.pytest(dryrun=True, mark='not recursive'):
#         print(cmd)
#         assert assert_that(cmd).is_equal_to(
#             'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /home/sheffler/sw/MambaForge/envs/mlb/bin/python -mpytest -v -m "not recursive" --collect-only > pytest_ipd_ci_test_run.log'
#         )
#         print(dict(log))

# def test_clitool_pytest_slurm_parallel():
#     tool = ipd.tools.ci.TestsTool()
#     cmd, result, log = zip(*tool.pytest(dryrun=True, parallel=4, slurm=True, which='test_foo test_bar'))
#     assert cmd == (
#         'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /home/sheffler/sw/MambaForge/envs/mlb/bin/python -mpytest -v -k "test_foo or test_bar" --collect-only > pytest_ipd_ci_test_run.log.noparallel.log',
#         'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /home/sheffler/sw/MambaForge/envs/mlb/bin/python -mpytest -v -k "not test_foo and not test_bar" --collect-only > pytest_ipd_ci_test_run.log.parallel.log',
#     )
#     a, b = log
#     assert a.selected == 2
#     assert b.deselected == 2

if __name__ == '__main__':
    main()

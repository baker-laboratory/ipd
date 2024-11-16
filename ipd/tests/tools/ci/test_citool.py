import os
import pytest
import rich

typer = pytest.importorskip('typer')
pytest.mark.skipif(int(typer.__version__.split('.')[1]) < 12, reason='ipd.sym.Helix breaks on numpy 2')

from typer.testing import CliRunner

import ipd

def main():
    # test_citool_update_library()
    test_clitool_pytest()
    # test_clitool_pytest_slurm_parallel()
    print('test clitool PASS')

def runipd(cmd):
    runner = CliRunner()
    tool = ipd.tools.IPDTool()
    result = runner.invoke(tool.__app__, cmd)
    if result.exit_code:
        rich.inspect(result)
    assert result.exit_code == 0
    return result.stdout

@pytest.mark.ci
def test_citool_update_library():
    runipd('ci update_library')

@pytest.mark.xfail
def test_that_fails():
    assert 0

def test_foo():
    pass

def test_bar():
    pass

os.chdir(f'{os.path.dirname(__file__)}/../../../..')

def test_clitool_pytest():
    out = runipd('ci tests pytest --cmdonly')
    assert out.strip(
    ) == 'cd TESTDIR && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. EXE --benchmark-disable --disable-warnings --cov --junitxml=junit.xml -o junit_family=legacy --durations=10 > pytest_ipd_ci_test_run.log'

    out = runipd("ci tests pytest --cmdonly --exe $exe --slurm --tee --gpu a4000")
    assert out.strip(
    ) == 'cd TESTDIR && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. EXE --benchmark-disable --disable-warnings --cov --junitxml=junit.xml -o junit_family=legacy --durations=10 2>&1 | tee pytest_ipd_ci_test_run.log'

    out = runipd(
        "ci tests pytest --cmdonly --exe $exe --slurm --parallel 4 --tee --which 'test_call_speed test_loss_grad'")
    assert out.strip(
    ) == """cd TESTDIR && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. EXE -k "not test_call_speed and not test_loss_grad" -n 4 --benchmark-disable --disable-warnings --cov --junitxml=junit.xml -o junit_family=legacy --durations=10 2>&1 | tee pytest_ipd_ci_test_run.log.par.log
cd TESTDIR && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. EXE -k "test_call_speed or test_loss_grad" --benchmark-disable --disable-warnings --cov --junitxml=junit2.xml -o junit_family=legacy --durations=10 2>&1 | tee pytest_ipd_ci_test_run.log.nopar.log"""

    # testdir = f'{ipd.projdir}/tests/crud'

# def test_clitool_pytest_slurm_parallel():
#     tool = ipd.tools.ci.TestsTool()
#     cmd, result, log = zip(*tool.pytest(dryrun=True, parallel=4, slurm=True, which='test_foo test_bar'))
#     assert cmd == (
#         'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /home/sheffler/sw/MambaForge/envs/mlb/bin/python -mpytest -v -k est_foo or test_bar" --collect-only > pytest_ipd_ci_test_run.log.noparallel.log',
#         'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. /home/sheffler/sw/MambaForge/envs/mlb/bin/python -mpytest -v -k ot test_foo and not test_bar" --collect-only > pytest_ipd_ci_test_run.log.parallel.log',
#     )
#     a, b = log
#     assert a.selected == 2
#     assert b.deselected == 2

if __name__ == '__main__':
    main()

import ipd
import sys

def test_run_tests_on_file():
    cmd = f'{sys.executable} {ipd.projdir}/tools/run_tests_on_file.py ipd {ipd.projdir}/tools/run_tests_on_file.py'
    ipd.dev.run(cmd)

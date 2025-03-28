import ipd
from ipd.tools.ipdtool import IPDTool
from ipd.tools import run_tests_on_file

class TestTool(IPDTool):

    def file(
        self,
        projects: list[str],
        path: str,
        pytest: bool = False,
    ):
        py = ipd.dev.run('which python')
        run_tests_on_file.main(projects, testfile=path, pytest=pytest, python=py)

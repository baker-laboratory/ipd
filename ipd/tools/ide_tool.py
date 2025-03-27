import ipd
from ipd.tools.ipdtool import IPDTool
from ipd.tools import run_tests_on_file

class IdeTool(IPDTool):

    def file(
        self,
        projects: list[str],
        path: str,
        pytest: bool = False,
    ):
        py = ipd.dev.run('/usr/bin/env python')
        run_tests_on_file.main(projects, testfile=path, pytest=pytest, python=py)

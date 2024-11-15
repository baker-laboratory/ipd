import ipd

class CodeTool(ipd.tools.IPDTool):
    def make_testfile(self, sourcefile, testfile):
        ipd.dev.make_testfile(sourcefile, testfile)

    def yapf(
        self,
        path: str = '[projname]',
        dryrun: bool = False,
        excludefile: str = '[gitroot].yapf_exclude',
        hashfile='[gitroot].yapf_hash',
        conffile='[gitroot]/pyproject.toml',
    ):
        """run yapf on all python files in path

        Args:
            path (str, optional): path to run yapf on. Defaults to '.'.
            dryrun (bool, optional): dry run. Defaults to False.
            exclude (str, optional): exclude file. Defaults to '.yapf_exclude'.
            hashfile (str, optional): hash file. Defaults to '.yapf_hash'.
        """
        ipd.dev.yapf_fast(path, dryrun, excludefile, hashfile, conffile)

class PyrightTool(CodeTool):
    def comment_type_errors(self, path: str = '.'):
        errors = ipd.dev.get_pyright_errors(path)
        ipd.dev.add_type_ignore_comments(errors)

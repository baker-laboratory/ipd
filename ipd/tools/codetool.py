from typing import Annotated
import typer
import ipd

class CodeTool(ipd.tools.IPDTool):
    def make_testfile(self, sourcefile, testfile):
        ipd.dev.make_testfile(sourcefile, testfile)

    def yapf(
        self,
        path: Annotated[str, typer.Argument()] = '[projname]',
        dryrun: bool = False,
        excludefile: str = '[gitroot].yapf_exclude',
        hashfile: str = '[gitroot].yapf_hash',
        conffile: str = '[gitroot]/pyproject.toml',
    ):
        """run yapf on all python files in path

        Args:
            path (str, optional): path to run yapf on. Defaults to '.'.
            dryrun (bool, optional): dry run. Defaults to False.
            exclude (str, optional): exclude file. Defaults to '.yapf_exclude'.
            hashfile (str, optional): hash file. Defaults to '.yapf_hash'.
            conffile (str, optional): config file. Defaults to 'pyproject.toml'.
        """
        cmd = 'yapf -ip --style {conffile} -m {" ".join(changed_files)}'
        result = ipd.dev.run_on_changed_files(cmd, path, dryrun, excludefile, hashfile, conffile)
        raise typer.Exit(code=int(result.files_modified))

    def pyright(
        self,
        path: Annotated[str, typer.Argument()] = '[projname]',
        add_type_ignore_comments: bool = False,
        dryrun: bool = False,
        excludefile: str = '[gitroot].pyright_exclude',
        hashfile: str = '[gitroot].pyright_hash',
        conffile: str = '[gitroot]/pyproject.toml',
    ):
        """run pyright on all python files in path

        Args:
            path (str, optional): path to run yapf on. Defaults to '.'.
            dryrun (bool, optional): dry run. Defaults to False.
            exclude (str, optional): exclude file. Defaults to '.yapf_exclude'.
            hashfile (str, optional): hash file. Defaults to '.yapf_hash'.
            conffile (str, optional): config file. Defaults to 'pyproject.toml'.
        """
        if add_type_ignore_comments:
            with ipd.dev.cd(ipd.dev.git_root()):
                path, = ipd.dev.substitute_project_vars(path)
                errors = ipd.dev.get_pyright_errors(path)
                ipd.dev.add_type_ignore_comments(errors)
                return
        cmd = 'pyright -p {conffile} {" ".join(changed_files)}'
        result = ipd.dev.run_on_changed_files(cmd, path, dryrun, excludefile, hashfile, conffile)
        raise typer.Exit(code=result.exitcode)

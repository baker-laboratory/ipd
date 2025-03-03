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

    def filter_python_output(self, path: Annotated[str, typer.Argument()]):
        # possible way to hangle kwargs
        # def main(item: list[str] = typer.Option(None, "--item", "-i", help="Key-value pairs (key=value)", allow_multiple=True)):
        # kwargs = {}
        # if item:
        # for i in item:
        # try:
        # key, value = i.split("=", 1)
        # kwargs[key] = value
        # except ValueError:
        # print(f"Invalid format for item: {i}. Expected key=value")
        with open(path) as inp:
            text = inp.read()
        with open(f'{path}.orig', 'w') as out:
            out.write(text)
        try:
            new = ipd.dev.filter_python_output(text, entrypoint='codetool', preset='ipd_boilerplate')
        except RuntimeError as e:
            with open(path, 'w') as out:
                out.write('ERROR WHEN RUNNING `ipd code filter_python_output <fname>`')
                out.write(e)
                raise typer.Exit()
        with open(path, 'w') as out:
            out.write(new)
            out.write('THIS FILE WAS FILTERED THRU `ipd code filter_python_output <fname>`')

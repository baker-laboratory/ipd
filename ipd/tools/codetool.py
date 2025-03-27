import os
from typing import Annotated
import typer
import ipd
from ipd.tools.ipdtool import IPDTool

class CodeTool(IPDTool):

    def make_testfile(self, sourcefile, testfile):
        ipd.dev.make_testfile(sourcefile, testfile)

    def clean_pycache(self, path: ipd.Path = ipd.Path('.')):
        os.system(fr'find {path} -name *.pyc -delete')

    def format_project(self,
                       path: Annotated[str, typer.Argument()] = '[projname]',
                       dryrun: bool = False,
                       excludefile: str = '[gitroot].yapf_exclude',
                       hashfile: str = '[gitroot].yapf_hash',
                       conffile: str = '[gitroot]/pyproject.toml'):
        """run ipd fromatting tool on all python files in path

        Args:
            path (str, optional): path to run ipdformat on. Defaults to '.'.
        """
        context = ipd.dev.OnlyChangedFiles('ipdformat', path, excludefile, hashfile, conffile)
        with context as fileinfo:
            ipd.kwcall(fileinfo, ipd.dev.format_files, fileinfo.changed_files, dryrun=dryrun)

    def format(self, path: Annotated[ipd.Path, typer.Argument(allow_dash=True)], dryrun: bool = False):
        """run ipd fromatting tool on all python files in path

        Args:
            path (str, optional): path to run ipdformat on. Defaults to '.'.
        """
        if path == '-': sys.stdout.write(ipd.dev.format_buffer(sys.stdin.read()))
        else: ipd.dev.format_files(path, dryrun=dryrun)
        ipd.dev.global_timer.report()

    def yapf(self,
             path: Annotated[str, typer.Argument()] = '[projname]',
             dryrun: bool = False,
             excludefile: str = '[gitroot].yapf_exclude',
             hashfile: str = '[gitroot].yapf_hash',
             conffile: str = '[gitroot]/pyproject.toml'):
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
        raise typer.Exit(code=int(bool(result.files_modified)))

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

    def filter_python_output(self, path: Annotated[str, typer.Argument()], keep_blank_lines=False):
        with open(path) as inp:
            text = inp.read()
        with open(f'{path}.orig', 'w') as out:
            out.write(text)
        try:
            new = ipd.dev.filter_python_output(text,
                                               entrypoint='codetool',
                                               preset='ipd_boilerplate',
                                               keep_blank_lines=keep_blank_lines)
        except RuntimeError as e:
            with open(path, 'w') as out:
                out.write('ERROR WHEN RUNNING `ipd code filter_python_output <fname>`')
                out.write(e)
                raise typer.Exit()
        with open(path, 'w') as out:
            out.write(new)
            out.write('\nTHIS FILE WAS FILTERED THRU `ipd code filter_python_output {path}`\n')

    def unique_errors(self, files: list[str]):
        text = ipd.dev.addreduce(ipd.Path(f).read_text() for f in files)
        try:
            result = ipd.dev.analyze_python_errors_log(text)
            print(result)
        except RuntimeError as e:
            print('ERROR WHEN RUNNING `ipd code alalyze_python_errors_log <fname>`')
            print(e)
            raise typer.Exit()

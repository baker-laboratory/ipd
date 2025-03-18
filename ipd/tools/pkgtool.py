import typer
from typing import Optional, Annotated

import ipd
from ipd.tools import IPDTool

class PkgTool(IPDTool):

    def cwd_package(self):
        return '/home/sheffler/ipd'

    def tarsource(
            self,
            source: Annotated[str, typer.Argument(help="Source directory to clone")],
            target: Annotated[str, typer.Argument(help="Target directory where to create the clone")],
            gitignore: Annotated[str, typer.Argument(help=".gitignore file")],
            max_size: Annotated[Optional[int], typer.Option(help="Maximum file size in bytes")] = None,
            min_size: Annotated[int, typer.Option(help="Minimum file size in bytes")] = 0,
            include_small_binary: Annotated[bool, typer.Option(help="Include small binary files")] = False,
            binary_threshold: Annotated[int,
                                        typer.Option(
                                            help="Size threshold for small binary files in bytes")] = 10240,
            verbose: Annotated[bool, typer.Option(help="Print verbose output")] = False):
        """
        Clone a directory structure but only copy text files (source code, config files, etc.).
        Binary files are skipped by default, with options to include small binary files.
        """
        target = ipd.dev.decompressed_fname(target)
        ipd.dev.CloneTextFiles(source,
                               target,
                               gitignore=gitignore,
                               max_size_bytes=max_size,
                               min_size_bytes=min_size,
                               include_small_binary=include_small_binary,
                               small_binary_threshold=binary_threshold,
                               verbose=verbose)
        # os.system(f'tar -czf {target}.tar.gz {target}')
        # os.system(f'rm -rf {target}')

class ConfigTool(PkgTool):

    def cmd_check(self):
        print('check_import')

class TestsTool(PkgTool):

    def cmd_check(self):
        print('check_import')

    def cmd_run(self):
        print('run_tests')

class ImportsTool(PkgTool):

    def cmd_check(self):
        print('check_import')

print(PkgTool.__descendants__)

if __name__ == '__main__':
    main()

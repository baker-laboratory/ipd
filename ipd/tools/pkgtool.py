import os
import typer
from typing import Annotated

import ipd
from ipd.tools import IPDTool

TO, TA = typer.Option, typer.Argument

class PkgTool(IPDTool):

    def cwd_package(self):
        return '/home/sheffler/ipd'

    def src_archive(
            self,
            source: Annotated[str, TA(help="Source directory to clone")],
            target: Annotated[str, TA(help="Target directory where to create the clone")],
            max_size: Annotated[int, TO(help="Maximum file size in bytes")] = 999999999,
            min_size: Annotated[int, TO(help="Minimum file size in bytes")] = 0,
            include_small_binary: Annotated[bool, TO(help="Include small binary files")] = False,
            binary_threshold: Annotated[int,
                                        TO(help="Size threshold for small binary files in bytes")] = 10240,
            verbose: Annotated[bool, TO(help="Print verbose output")] = False):
        """
        Clone a directory structure but only copy text files (source code, config files, etc.).
        Binary files are skipped by default, with options to include small binary files.
        """
        os.system(f'rm {target}')
        target, origtarget = f'/tmp/{target}', target
        target, compressed = ipd.dev.decompressed_fname(target), target
        assert not target.endswith('.tar.gz') and not target.endswith('.zip')
        files = ipd.dev.project_files()
        _ = ipd.dev.CloneTextFiles(source,
                                   target,
                                   max_size_bytes=max_size,
                                   min_size_bytes=min_size,
                                   include_small_binary=include_small_binary,
                                   small_binary_threshold=binary_threshold,
                                   verbose=verbose,
                                   filelist=files)
        # os.system(f'mkdir __tmp__')
        # os.system(f'mv {target}/* __tmp__')
        # os.system(f'mv __tmp__ {target}/ipd')
        if compressed.endswith('.tar.gz'):
            os.system(f'tar -czf {target}.tar.gz {target}')
            os.system(f'rm -rf {target}')
        if compressed.endswith('.zip'):
            os.system(f'zip -r {target}.zip {target}')
            os.system(f'rm -rf {target}')
        os.system(f'mv {target} {origtarget}')

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

# print(PkgTool.__descendants__)

if __name__ == '__main__':
    main()

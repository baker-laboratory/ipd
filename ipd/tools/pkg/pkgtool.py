# import os
import sys

from ipd.dev.cli import CliBase

def main():
    tool = PkgTool()
    sys.argv += 'imports check'.split()
    tool.run()
    print('check_package_structure.py DONE')

class PkgTool(CliBase):

    def cwd_package(self):
        return '/home/sheffler/ipd'

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

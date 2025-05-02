import os
import sys
import evn

with evn.cd_project_root() as project_exists:
    using_local_build = False
    if project_exists and os.path.exists('_build'):
        assert os.path.exists('pyproject.toml')
        os.system('doit build')
        try:
            sys.path.append(f'_build/py3{sys.version_info.minor}')
            print(f'_build/py3{sys.version_info.minor}')
            from _detect_formatted_blocks import *  # type: ignore
            from _token_column_format import *  # type: ignore
            using_local_build = True
        except ImportError:
            pass
        finally:
            sys.path.pop(0)  # Remove the build path so it doesn't interfere with import
    if not using_local_build:
        from evn.format._detect_formatted_blocks import *  # type: ignore
        from evn.format._token_column_format import *  # type: ignore

from evn.format.formatter import *

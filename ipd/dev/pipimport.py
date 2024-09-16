import importlib
import subprocess
import sys

def pipimport(modulename, pipname=None):
    pipname = pipname or modulename
    import importlib
    try:
        return importlib.import_module(modulename)
    except (ValueError, ModuleNotFoundError):
        subprocess.check_call(f'{sys.executable} -mpip install {pipname}'.split())
        try:
            return importlib.import_module(modulename)
        except (ValueError, ModuleNotFoundError):
            subprocess.check_call(f'{sys.executable} -mpip install --user {pipname}'.split())
            return importlib.import_module(modulename)

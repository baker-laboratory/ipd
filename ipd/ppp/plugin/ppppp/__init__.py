import contextlib

with contextlib.suppress(ImportError):
    import pymol, sys, os
    from pymol.plugins import addmenuitemqt
    ipd_path = os.path.realpath(os.path.dirname(__file__) + '/../../../..')
    if ipd_path not in sys.path: sys.path.append(ipd_path)
    sys.path.append(f'/home/sheffler/project/ppp/lib/python3.{sys.version_info.minor}/site-packages')
    from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin import *

    def __init_plugin__(app=None):
        sys.stderr.write('init pymol plugin')
        path = os.path.dirname(__file__)
        addmenuitemqt(f'Pretty Protein Project Pymol Plugin {path}', run)

import contextlib

with contextlib.suppress(ImportError):
    import os
    import sys

    from pymol.plugins import addmenuitemqt
    ipd_path = os.path.realpath(os.path.dirname(__file__) + '/../../../..')
    if ipd_path not in sys.path: sys.path.append(ipd_path)
    newpath = f'/home/sheffler/project/ppp/lib/python3.{sys.version_info.minor}/site-packages'
    sys.path.append(newpath)
    print('PATH ADDED TO sys.path:', newpath)
    from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin import *

    def __init_plugin__(app=None):
        sys.stderr.write('init pymol plugin')
        os.path.dirname(__file__)
        addmenuitemqt('Pretty Protein Project Pymol Plugin', run)

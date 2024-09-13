import contextlib

with contextlib.suppress(ImportError):
    import pymol, sys, os
    from pymol.plugins import addmenuitemqt
    ipd_path = os.path.realpath(os.path.dirname(__file__) + '/../../../..')
    if ipd_path not in sys.path: sys.path.append(ipd_path)

    def __init_plugin__(app=None):
        path = os.path.dirname(__file__)
        from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin import run
        addmenuitemqt(f'Pretty Protein Project Pymol Plugin {path}', run)

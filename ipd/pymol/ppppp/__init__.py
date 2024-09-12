import contextlib

# def print_hello():
#     print("PPPPP",flush=True)

with contextlib.suppress(ImportError):
    import pymol, sys, os
    ipd_path = os.path.realpath(os.path.dirname(__file__) + '/..')
    if ipd_path not in sys.path: sys.path.append(ipd_path)

    from ipd.pymol.ppppp.prettier_protein_project_pymol_plugin import run_ppppp_gui

    #     def __init_plugin__(app=None):
    #         with open('/home/sheffler/tmp/PYMOL_OUT','w') as out:
    #             out.write('ADD MENU ITEM\n')
    #         from pymol.plugins import addmenuitemqt
    #         addmenuitemqt('Pretty Protein Project PyMOL Plugin', print_hello)

    def __init_plugin__(app=None):
        from pymol.plugins import addmenuitemqt as addmenuitem
        addmenuitem('Pretty Protein Project Pymol Plugin', run_ppppp_gui)

import ipd
from ipd.viz.pymol_viz import lazy_register

def show_body_pymol(body, name='body'):
    ipd.viz.show_atoms_pymol(body.positioned_atoms, name)

def show_symbody_pymol(symbody, name='symbody'):
    origids = symbody.asu.atoms.chain_id
    all_pymol_chains = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz")
    uniqids = ipd.dev.UniqueIDs(all_pymol_chains)
    for i, body in enumerate(symbody.bodies):
        body.atoms.chain_id = uniqids(body.atoms.chain_id, reset=True)
        show_body_pymol(body)
    symbody.asu.atoms.chain_id = origids

@lazy_register('Body')
def regester_body():

    @ipd.viz.pymol_load.register(ipd.atom.Body)
    def pymol_viz_body(body, name, state, **kw):
        show_body_pymol(body, name, **kw)

@lazy_register('SymBody')
def regester_symbody():

    @ipd.viz.pymol_load.register(ipd.atom.SymBody)
    def pymol_viz_symbody(symbody, name, state, **kw):
        show_symbody_pymol(symbody, name, **kw)

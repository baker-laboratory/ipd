import lark
import ipd

with ipd.dev.open_package_file('sel/pymol_selection_algebra.lark') as pymol_grammar:
    pymol_selection_parser = lark.Lark(pymol_grammar)

# ipd.sel.pymol
def pymol(sel):
    pymol_selection_parser.parse(sel)

class AtomTransformer(lark.Transformer):

    def __init__(self, atoms):
        self.atoms = atoms

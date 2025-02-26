import contextlib
import ipd

with contextlib.suppress(ImportError):
    import lark

    with ipd.dev.open_package_file('sel/pymol_selection_algebra.lark') as pymol_grammar:
        pymol_selection_parser = lark.Lark(pymol_grammar)  #, parser='lalr')

    # ipd.sel.pymol
    def pymol(sel):
        pymol_selection_parser.parse(sel)

    class AtomTransformer(lark.Transformer):

        def __init__(self, atoms):
            self.atoms = atoms

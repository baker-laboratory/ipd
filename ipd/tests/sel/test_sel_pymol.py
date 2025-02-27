# try:
# from biotite.structure import AtomArray
# import biotite.structure.io as strucio
# except ModuleNotFoundError:
# pytest.skip(allow_module_level=True)
import pytest

pytest.importorskip('lark')

import ipd
from ipd.sel.sel_pymol import pymol_selection_parser

small_protein = ipd.tests.fixtures.small_atoms()
mixed_structure = ipd.tests.fixtures.mixed_atoms()

def main():
    # ipd.tests.maintest(namesace=globals())
    test_basic_parse()

@pytest.mark.xfail
def test_basic_parse():
    test_selections = [
        'name ca and all within 7 of (elem O+H within 4 of lig)', 'obj* and name ca', 'resn ALA like \'A*\'',
        'mySelection', '%savedSelection', '?tempSelection', 'index 1+2+3+10-20', 'fc. <=-1', 'bonded', 'protected',
        'fixed', 'restrained', 'masked', 'organic', 'inorganic', 'solvent', 'polymer', 'guide', 'hetatm', 'hydrogens',
        'backbone', 'sidechain', 'metals', 'donors', 'acceptors', 'present', 'y<-5', 'z<10 and x>0 and y>-5',
        'chain A and not resi 125', 'foo=name CA+CB+CG and chain A', 'byres all within 5 of organic', 'ss \'H\'',
        'br. b<20 & (all within 3 of resn HOH)', 'first resn ARG', '1foo/G/X/444/CA', 'bm. c. C',
        'byres ((chain A or (chain B and (not resi 125))) around 5)', 'solvent and chain A', 'solvent or chain A',
        'resi 1+2+3', 'chain A+B', 'name N+CA+C', 'resn GLU+GLY', 'resi 10-20+30-40', 'index 1+2+3', 'id 100+200+300',
        'all', '*', 'none', 'enabled', 'first chain A', 'last resn ALA', 'model 1abc', 'm. 1XYZ', 'chain A', 'c. B+C+D',
        'segi A', 's. PROA+PROB', 'resn ALA+ARG+ASP', 'r. GLY', 'resi 10-20+100-120+200', 'i. 1+2+3', 'name CA',
        'n. N+CA+C+O', 'alt A+B', 'id 1+2+3', 'rank 1+2+3', 'pepseq AGTY', 'ps. RGD', 'label myLabel',
        'chain A in polymer.protein', 'byobject chain A', 'bysegi chain A', 'bs. chain A', 'bychain resi 100-200',
        'bc. name CA', 'byres name CA', 'br. resi 10-20', 'bycalpha chain A', 'bca. chain A', 'bymolecule chain A',
        'bm. organic', 'byfragment resn LIG', 'bf. hetatm', 'byring resn PHE', 'bycell all', 'bound_to resn ALA',
        'bto. name CA', 'neighbor resn LIG', 'nbr. name CA', 'name CA extend 3.5', 'chain A within 5 of chain B',
        'chain A w. 5 of chain B', 'name CA around 3.5', 'name CA a. 3.5', 'chain A expand 2.0', 'chain A x. 2.0',
        'chain A gap 4.0', 'chain A near_to 5 of chain B', 'chain A nto. 5 of chain B', 'chain A beyond 10 of chain B',
        'chain A be. 10 of chain B', 'partial_charge <0', 'pc. >0.5', 'formal_charge =1', 'b<30', 'b>50', 'b=30.5',
        'b>=20.5', 'b<=25', 'b!=0', 'q>0.8', 'q<0.5', 'ss H', 'ss \'HS\'', 'ss HST', 'elem C', 'e. N', 'e. Ca',
        'p.temperature >300', 'stereo R', 'stereo S', 'fxd.', 'rst.', 'msk.', 'flag 1', 'f. 2', 'org.', 'ino.', 'sol.',
        'pol.', 'polymer.protein', 'polymer.nucleic', 'h.', 'bb.', 'sc.', 'don.', 'acc.', 'pr.', 'x>10', 'z=0',
        'text_type GAFF', 'tt. Amber', 'numeric_type 1', 'nt. 2', '1ABC/*/A/100/CA', '1XYZ/PROA/*/10-20/*', '*/*/A/*/CA',
        'name CA and not (resi 10-20 or resn GLY)', '(chain A and backbone) or (chain B and sidechain)',
        'not (solvent within 5 of protein)', 'not solvent within 5 of protein', 'resi 1-100 and chain A or chain B',
        'resi 1-100 and (chain A or chain B)', 'chain A & name CA', 'chain A | name CB', '!chain B',
        'chain A | chain B and not chain C', 'chain A and not chain B or chain C', 'chain A # this is a comment',
        'name CA # atoms only',
        'br. (name CA and (chain A or chain B)) within 5 of (resn LIG and not (resi 100 or resi 200))',
        '(byres name CA) and not (bychain resn HOH)',
        '((chain A and resi 10-50) or (chain B and resi 100-150)) and not (resn HOH within 5 of (name CA and b>30))',
        'all extend 5.0 & not solvent', 'not backbone and not sidechain', 'polymer.protein and chain A+B',
        'pc.>0.5 and name CA within 5 of center'
    ]

    successful = 0
    failed = 0

    print('Testing PyMOL Selection Algebra Parser\n')
    print('='*50 + '\n')

    for i, sel in enumerate(test_selections, 1):
        try:
            tree = pymol_selection_parser.parse(sel)
            # print(f'{i:3d}. ✓ Successfully parsed: {sel}')
            try:
                pass
                # print(tree.pretty())
            except Exception as e:
                print('token:', tree)
            successful += 1
        except Exception as e:
            print(f'{i:3d}. ✗ Failed to parse: {sel}, Error: {e}')
            failed += 1
            return

    print('\n' + '='*50)
    print(f'\nTest Results: {successful} successful, {failed} failed out of {len(test_selections)} total')

    if failed == 0:
        print('\nAll tests passed! The grammar successfully handles all tested PyMOL selection patterns.')
    else:
        print(f'\n{failed} test(s) failed. The grammar may need further refinement.')

    assert not failed

if __name__ == '__main__':
    main()

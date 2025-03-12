import pytest
import ipd

bs = pytest.importorskip('biotite.structure')

TEST_PDBS = [
    '3sne',
    '1dxh',
    '1n0e',
    '1wa3',
    '1a2n',
    '1n0e',
    '1bfr',
    '1g5q'  #, '3woc', '7abl', '2tbv'
]

config_test = ipd.Bunch(
    re_only=[],
    re_exclude=[],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
        dryrun=False,
    )

# def test_assembly_simple():
# asm = ipd.atom.assembly('1qys')

def make_assembly_pdb_tests():
    for pdb in TEST_PDBS:

        class TestAssembly():

            def __init__(self):
                self.asm = ipd.atom.assembly(pdb, min_chain_atoms=50)
                self.asm = ipd.atom.assembly(pdb, min_chain_atoms=0)

            def test_construction(self):
                print(self.asm)

                assert 0

            def test_random_chain(self):
                assert 0

            def test_neighborhood(self):
                assert 0

        globals()[f'TestAssembly_{pdb.upper()}_'] = TestAssembly

make_assembly_pdb_tests()

if __name__ == '__main__':
    main()

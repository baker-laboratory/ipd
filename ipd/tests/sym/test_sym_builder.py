import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_build_from_components():
    atoms1 = ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/L2_D1_C3_Apo.pdb'))
    atoms2 = ipd.pdb.readatoms(ipd.dev.package_testdata_path('pdb/L2_D1_C5.pdb'))

    ipd.sym.build_from_components(atoms1, atoms2)
    assert 0

if __name__ == '__main__':
    main()

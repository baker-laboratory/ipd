import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_get_pdb_symmetry():
    assert ipd.pdb.get_pdb_symmetry('1bfr')['polymer_entity_instance_count'] == 24

if __name__ == '__main__':
    main()

import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_pdb_info():
    info = ipd.pdb.info('2tbv')
    assert info.entry.id == '2TBV'
    assert info.rcsb_entry_info.assembly_count == 1

def test_pdb_info_assembly():
    info = ipd.pdb.info('2tbv', assembly=1)
    ic(info.keys())
    assert info.rcsb_struct_symmetry[0].symbol == 'I'
    assert info.pdbx_struct_assembly_gen[0].asym_id_list == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    assert info.pdbx_struct_assembly_gen[0].oper_expression == '(1-60)'
    assert len(info.pdbx_struct_oper_list) == 62
    assert info.pdbx_struct_oper_list[0].name == '1_555'
    assert info.pdbx_struct_oper_list[0].matrix11 == 1

def test_pdb_info_all_assemblies():
    assert len(ipd.pdb.info('2tbv', assembly='all')) == 1

if __name__ == '__main__':
    main()

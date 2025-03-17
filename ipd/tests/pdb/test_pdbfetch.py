import os
import pytest
import ipd

TEST_PDB_CODES = ['1crn', '1a3n', '2ci2']

def main():
    ipd.tests.maintest(namespace=globals())

@ipd.dev.timed
def test_pdb_info():
    info = ipd.pdb.rcsbinfo('2tbv')
    assert info.entry.id == '2TBV'
    assert info.rcsb_entry_info.assembly_count == 1
    info2 = ipd.pdb.rcsbinfo('1qys 2tbv')
    assert info2['2tbv'] == info
    assert info2['1qys'].entry.id == '1QYS'

@ipd.dev.timed
def test_pdb_info_assembly():
    info = ipd.pdb.rcsbinfo('2tbv', assembly=1)
    assert info.rcsb_struct_symmetry[0].symbol == 'I'
    assert info.pdbx_struct_assembly_gen[0].asym_id_list == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    assert info.pdbx_struct_assembly_gen[0].oper_expression == '(1-60)'
    assert len(info.pdbx_struct_oper_list) == 62
    assert info.pdbx_struct_oper_list[0].name == '1_555'
    assert info.pdbx_struct_oper_list[0].matrix11 == 1

@ipd.dev.timed
def test_pdb_info_all_assemblies():
    assert len(ipd.pdb.rcsbinfo('2tbv', assembly='all')) == 1

@ipd.dev.timed
def test_pdb_info_speed():
    with ipd.dev.Timer() as t:
        for i in range(100):
            # dat = requests.get(f'https://data.rcsb.org/rest/v1/core/entry/2TBV').json()
            ipd.pdb.rcsbinfo('2tbv')
    assert t.elapsed() < 10

@ipd.dev.timed
def test_pdb_sym_annotation():
    symanno_1hv4 = ipd.pdb.sym_annotation('1hv4')
    assert symanno_1hv4.sym == tuple('C2 D2 C2 D2'.split())
    symanno = ipd.pdb.sym_annotation('1hv4 1out 1ql2')
    assert symanno.sym == ('C2', 'D2', 'C2', 'D2', 'C2', 'D2', 'H')

@pytest.mark.parametrize('pdb_code', TEST_PDB_CODES)
def test_download_bcif(pdb_code, tmpdir):
    output_file = tmpdir / f"{pdb_code}.bcif.gz"
    ipd.pdb.download_bcif(pdb_code, str(output_file))
    assert output_file.exists(), f"File {output_file} was not created"
    assert os.stat(output_file).st_size > 0, f"File {output_file} is empty"

def test_invalid_pdb_code(tmpdir):
    pdb_code = 'xxxx'  # Invalid code
    output_file = tmpdir / f"{pdb_code}.bcif.gz"

    with pytest.raises(Exception, match="Failed to download"):
        ipd.pdb.download_bcif(pdb_code, str(output_file))

if __name__ == '__main__':
    main()

import pytest

import ipd

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
    )

def print_tform_pipeline_datahub_data(data):
    print('example_id', data['example_id'])
    print('path', data['path'])
    print('pdb_id', data['pdb_id'])
    print('assembly_id', data['assembly_id'])
    print('query_pn_unit_iids', data['query_pn_unit_iids'])
    print('extra_info', data['extra_info'])
    print('atom_array', data['atom_array'].coord.shape)
    print('atom_array_stack', data['atom_array_stack'].coord.shape)
    print('chain_info', data['chain_info'].keys())
    print('ligand_info', data['ligand_info'])
    print('metadata', data['metadata'])

@pytest.mark.skip
def test_symmetric_crop():
    # pytest.importorskip('datahub')
    # path = ipd.dev.package_testdata_path('6u9d_Q_1.pickle.gz')
    # data = ipd.dev.load(path).data
    # atoms = data['atom_array']
    # chains = ipd.atom.chain_dict(atoms)
    # print(chains)
    # atoms = ipd.atom.remove_nan_atoms(atoms)
    # atoms = ipd.atom.remove_garbage_residues(atoms)
    # atoms = ipd.atom.remove_nonprotein(atoms)
    # atoms = ipd.atom.split_chains(atoms)
    # ipd.atom.dump(atoms[0], '/tmp/test1.cif')
    # ipd.atom.dump(atoms[1], '/tmp/test2.cif')
    # assert 0
    # ipd.atom.dump(atoms, '/home/sheffler/tmp/6u9d_Q_1.cif')
    # ic(data.keys())
    # ic(data['assembly_id'])
    # ic(data['query_pn_unit_iids'])
    # ic(data['atom_array'].coord.shape)
    # ic(data['atom_array_stack'].coord.shape)
    # print([str(c) for c in data['chain_info'].keys()])
    # ic(len(data['chain_info'].keys()))
    # ic(len(np.unique(data['atom_array'].chain_id)))
    # ipd.showme(atoms)
    atoms = ipd.atom.load('6u9d', assembly='1')
    comps = ipd.atom.find_components_by_seqaln_rmsfit(atoms)
    comps.print_intermediates()
    ic(comps)
    assert 0

if __name__ == '__main__':
    main()

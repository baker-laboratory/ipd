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

def test_symmetric_crop():
    pytest.importorskip('datahub')
    path = ipd.dev.package_testdata_path('datahub_crop_example.pickle.gz')
    data = ipd.dev.load(path)
    print(data.keys())

if __name__ == '__main__':
    main()

import glob
import os

import requests

import ipd

def all_test_bcif(path=None):
    path = path or ipd.dev.package_testdata_path('pdb')
    bcif = set(os.path.basename(f)[:-8] for f in glob.glob(f'{path}/*.bcif.gz'))
    return bcif

def download_test_pdbs(pdbs, path=None, overwrite=False):
    path = path or ipd.dev.package_testdata_path('pdb')
    existing, pdbs = all_test_bcif(path), set(pdbs)
    if not overwrite: pdbs -= existing
    if pdbs:
        from biotite.database.rcsb import fetch
        fetch(pdbs, 'bcif', path, verbose=True)
        ipd.dev.run(f'gzip {path}/*.bcif')
    return pdbs

def get_pdb_symmetry(pdb_id):
    pdb_id = pdb_id.strip().lower()
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve data for PDB ID {pdb_id}. Status code: {response.status_code}")
    data = response.json()
    symmetry = {}
    if 'rcsb_entry_info' in data:
        entry_info = data['rcsb_entry_info']
        if 'space_group' in entry_info:
            symmetry['space_group'] = entry_info['space_group']
        if 'symmetry' in entry_info:
            symmetry['symmetry'] = entry_info['symmetry']

    if 'cell' in data:
        cell_data = data['cell']
        cell_info = {}
        for key in ['length_a', 'length_b', 'length_c', 'angle_alpha', 'angle_beta', 'angle_gamma']:
            if key in cell_data:
                cell_info[key] = cell_data[key]

        if cell_info:
            symmetry['cell_dimensions'] = cell_info

    if 'rcsb_entry_info' in data and 'deposited_polymer_entity_instance_count' in data['rcsb_entry_info']:
        symmetry['polymer_entity_instance_count'] = data['rcsb_entry_info']['deposited_polymer_entity_instance_count']

    if 'rcsb_assembly_info' in data:
        assembly_info = data['rcsb_assembly_info']
        if assembly_info:
            symmetry['assembly_info'] = assembly_info

    return symmetry

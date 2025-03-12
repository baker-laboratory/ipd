import functools
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

@functools.lru_cache
def rcsb_get(path, retries=3):
    url = f'https://data.rcsb.org/rest/v1/core/{path}'
    for _ in range(retries):
        response = requests.get(url)
        if response.status_code == 200: return response.json()
    raise ValueError(f'cant fetch rcsb info: {url}, tried {retries} times')

@ipd.dev.iterize_on_first_param(basetype=str, splitstr=True, asbunch=True)
def rcsbinfo(pdb, assembly=None):
    pdb = pdb.upper()
    if assembly == 'all':
        return [rcsbinfo(pdb, assembly=i) for i in range(1, assembly_count(pdb) + 1)]
    elif assembly is not None:
        data = rcsb_get(f'assembly/{pdb}/{assembly}')
    else:
        data = rcsb_get(f'entry/{pdb}')
    return ipd.bunchify(data)

def assembly_count(pdb):
    return rcsbinfo(pdb).rcsb_entry_info.assembly_count

def sym_annotation(pdb):
    infodict = rcsbinfo(pdb, assembly='all')
    if not isinstance(infodict, dict): infodict = {pdb: infodict}  # single value
    vals = []
    for pdb, infolist in infodict.items():
        for iasm, asm in enumerate(infolist):
            id, candasm = asm.pdbx_struct_assembly.id, asm.pdbx_struct_assembly.rcsb_candidate_assembly
            assert id == asm.rcsb_assembly_info.assembly_id
            if 'rcsb_struct_symmetry' not in asm: continue
            for isym, ssym in enumerate(map(ipd.Bunch, asm.rcsb_struct_symmetry)):
                val = (pdb, iasm, id, candasm, isym, *ssym.fzf('symbol type stoichio olig_st'))
                vals.append(val)
    names = 'pdb iasm id id2 isym sym type stoi oligo'
    result = ipd.Bunch(zip(names.split(), zip(*vals)))
    return result

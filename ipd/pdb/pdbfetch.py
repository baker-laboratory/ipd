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
def info(pdb, assembly=None):
    pdb = pdb.upper()
    if assembly == 'all':
        return [info(pdb, assembly=i) for i in range(1, assembly_count(pdb) + 1)]
    elif assembly is not None:
        dat = requests.get(f'https://data.rcsb.org/rest/v1/core/assembly/{pdb}/{assembly}').json()
    else:
        dat = requests.get(f'https://data.rcsb.org/rest/v1/core/entry/{pdb}').json()
    return ipd.bunchify(dat)

def assembly_count(pdb):
    return info(pdb).rcsb_entry_info.assembly_count

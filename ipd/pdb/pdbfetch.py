import glob
import os

# import requests
import httpx

import ipd

def all_test_bcif(path=None):
    path = path or ipd.dev.package_testdata_path('pdb')
    bcif = set(os.path.basename(f)[:-8] for f in glob.glob(f'{path}/*.bcif.gz'))
    return bcif

@ipd.dev.timed
def download_test_pdbs(pdbs, path=None, overwrite=False):
    path = path or ipd.dev.package_testdata_path('pdb')
    existing, pdbs = all_test_bcif(path), set(pdbs)
    if not overwrite: pdbs -= existing
    for pdb in pdbs:
        download_bcif(pdb, path, verbose=True)
        os.system(f'gzip {path}/*.bcif')
    return pdbs

@ipd.dev.timed
def download_bcif(pdb_code: str, output_file: str, verbose: bool = True) -> None:
    """
    Downloads a .bcif.gz file from RCSB for a given PDB code.

    Args:
        pdb_code (str): The 4-letter PDB code.
        output_file (str): The output file path.

    Raises:
        Exception: If the request fails.
    """
    url = f"https://models.rcsb.org/{pdb_code.upper()}.bcif.gz"
    if os.path.isdir(output_file): output_file = os.path.join(output_file, f"{pdb_code}.bcif.gz")
    # response = requests.get(url, stream=True)
    response = httpx.get(url)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        # if not output_file.endswith('.bcif.gz'): output_file += '.bcif.gz'
        # with open(output_file, 'wb') as f:
        # for chunk in response.iter_content(chunk_size=8192):
        # f.write(chunk)
        if verbose: print(f"Downloaded {pdb_code}.bcif.gz to {output_file}")
    else:
        raise RuntimeError(f"Failed to download {url} (status code: {response.status_code})")

@ipd.dev.safe_lru_cache
@ipd.dev.timed
def rcsb_get(path, retries=3):
    url = f'https://data.rcsb.org/rest/v1/core/{path}'
    for _ in range(retries):
        response = httpx.get(url)
        if response.status_code == 200: return response.json()
    raise ValueError(f'cant fetch rcsb info: {url}, tried {retries} times')

@ipd.dev.timed
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

@ipd.dev.timed
def assembly_count(pdb):
    return rcsbinfo(pdb).rcsb_entry_info.assembly_count

@ipd.dev.timed
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

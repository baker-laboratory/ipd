import json

import pytest

import ipd

@pytest.mark.skip
def test_pdb_concat(three_PDBFiles):
    print(three_PDBFiles[0].df)

    # dpdb      idp,                code, nres, othes meta
    # dres      idr,     idp,       ri, rn, ch, cen, bbcoord, nbrs?, ONLY PEPTIDE?
    # datom     ida,     idr, idp   an, rn, el, hyb, degree?, charges, vol, stub?? LIG AND SC?
    # dresres   idrr,    idr, idr,            cendist, xform, bbdistmat?
    # dresatom  idra,    idr, ida,            local_atom_coord
    # datomatom idaa,    ida, ida,            bondtype, bonddegree

def inspect_mpnn_jsonl():
    for l in open("mpnn_json_lines.jsonl", "r"):
        j = json.loads(l)
        for k, v in j.items():
            print(k, type(v), len(v))
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    print("   ", k2, type(v2), len(v2))

def main():
    # pat = os.path.join(ipd.dev.package_testdata_path, 'pdb/*.pdb1.gz')
    # ipd.tests.save_test_data(ipd.pdb.load_pdbs(pat), 'pdb/three_PDBFiles.pickle')

    pdbfiles = ipd.tests.fixtures.three_PDBFiles()
    test_pdb_concat(pdbfiles)
    # inspect_mpnn_jsonl()

if __name__ == "__main__":
    main()

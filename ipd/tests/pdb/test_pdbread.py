import os

import pandas as pd

import ipd

# ic.configureOutput(includeContext=True, contextAbsPath=False)

def main():
    from ipd.tests import fixtures as f

    # assert 0, 'MAIN'
    test_pdbread(f.pdbfname(), f.pdbcontents())
    test_load_pdbs(f.pdbfnames())
    test_find_pdb_files()

    ic("TEST_PDBREAD DONE")  # type: ignore

def firstlines(s, num, skip):
    count = 0
    for line in s.splitlines():
        if not line.startswith("ATOM"):
            continue
        count += 1
        if count > skip:
            print(line)
        if count == num + skip:
            break

# COLUMNS        DATA TYPE       CONTENTS
# --------------------------------------------------------------------------------
#  1 -  6        Record name     "ATOM  "
#  7 - 11        Integer         Atom serial number.
# 13 - 16        Atom            Atom name.
# 17             Character       Alternate location indicator.
# 18 - 20        Residue name    Residue name.
# 22             Character       Chain identifier.
# 23 - 26        Integer         Residue sequence number.
# 27             AChar           Code for insertion of residues.
# 31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
# 39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
# 47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
# 55 - 60        Real(6.2)       Occupancy.
# 61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
# 73 - 76        LString(4)      Segment identifier, left-justified.
# 77 - 78        LString(2)      Element symbol, right-justified.
# 79 - 80        LString(2)      Charge on the atom.

def test_pdbread(pdbfname, pdbcontents):
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    foo = (
        #    5    10   15   20   25   30   35   40   45   50   55   60   65   70   75   80
        #    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
        # hhhhhhiiiii_aaaaLrrr_CiiiiI___xxxxxxxxyyyyyyyyzzzzzzzzoooooobbbbbb      ssssEEcc
        "HETATM12345 ATOM RES C 1234   1236.8572215.5813376.721440.50547.32      SEGIPBCH\n" +
        "ATOM1234567 ATOM RES C 1234   1236.8572215.5813376.721440.50547.32      SEGIPBCH\n")
    pdb = ipd.pdb.readpdb(foo)
    assert all(pdb.df.columns == ["het", "ai", "an", "rn", "ch", "ri", "x", "y", "z", "occ", "bfac", "elem", "mdl"])
    assert pdb.df.shape == (2, 13)
    assert all(pdb.df.ai == (12345, 1234567))

    # num, skip = 1, 0
    # firstlines(pdbcontents, num, skip)

    pdb2 = ipd.pdb.readpdb(pdbcontents)
    pdb1 = ipd.pdb.readpdb(pdbfname)
    assert all(pdb1.df == pdb2.df)
    assert pdb1.cryst1 == pdb2.cryst1
    assert pdb1.seq == pdb2.seq
    assert pdb1.fname == pdbfname

    assert pdb1.seq == "ELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVA"
    # print(pdbcontents)
    # for c in pdb1.df.columns:
    # print(pdb1.df[c][60])
    types = [type(_) for _ in pdb1.df.loc[0]]
    for i in range(len(pdb1.df)):
        assert types == [type(_) for _ in pdb1.df.loc[i]]

def test_load_pdbs(pdbfnames):
    seqs = [
        "ELTPAVTTYKLVINGKTLKGETTTKAVDAETAEKAFKQYANDNGVDGVWTYDDATKTFTVTEMVTEVPVA",
        "DIQVQVNIDDNGKNFDYTYTVTTESELQKVLNELZDYIKKQGAKRVRISITARTKKEAEKFAAILIKVFAELGYNDINVTFDGDTVTVEGQL",
    ]
    pdbs = ipd.pdb.load_pdbs(pdbfnames, cache=False, pbar=False)
    assert set(pdbs.keys()) == set(pdbfnames)
    for i, fname in enumerate(pdbs):
        assert pdbs[fname].seqhet == seqs[i]

def test_find_pdb_files():
    pat = ipd.dev.package_testdata_path("pdb/*.pdb1.gz")
    files = ipd.pdb.find_pdb_files(pat)
    found = set(os.path.basename(f) for f in files)
    check = {"1qys.pdb1.gz", "1coi.pdb1.gz", "1pgx.pdb1.gz"}
    assert check.issubset(found)

if __name__ == "__main__":
    main()

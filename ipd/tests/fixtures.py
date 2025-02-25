import contextlib
import numpy as np
from ipd.dev import load_package_data, package_testdata_path
from ipd.pdb import readpdb

def pdbcontents():
    return load_package_data("tests/pdb/1pgx.pdb1.gz")

def pdbfname():
    return package_testdata_path("pdb/1pgx.pdb1.gz")

def pdbfnames():
    return [package_testdata_path("pdb/1pgx.pdb1.gz"), package_testdata_path("pdb/1qys.pdb1.gz")]

def three_PDBFiles():
    return [readpdb(package_testdata_path(f"pdb/{code}.pdb1.gz")) for code in ["3asl", "1pgx", "1coi"]]
    # return [load_package_data(ftests/'pdb/{code}.pdb1.gz.pickle') for code in ['3asl', '1pgx', '1coi']]

def pdbfile():
    return readpdb(package_testdata_path("pdb/3asl.pdb1.gz"))
    # return load_package_data('tests/pdb/3asl.pdb1.gz.pickle')

def pdb1pgx():
    return readpdb(package_testdata_path("pdb/1pgx.pdb1.gz"))
    # return load_package_data('tests/pdb/1pgx.pdb1.gz.pickle')

def pdb1coi():
    return readpdb(package_testdata_path("pdb/1coi.pdb1.gz"))
    # return load_package_data('tests/pdb/1coi.pdb1.gz.pickle')

def pdb1qys():
    return readpdb(package_testdata_path("pdb/1qys.pdb1.gz"))
    # return load_package_data('tests/pdb/1qys.pdb1.gz.pickle')

def ncac():
    return readpdb(pdbfname()).ncac()
    # pdb = pdb.subset(atomnames=['N', 'CA', 'C'], chains=['A'])
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T.reshape(-1, 3, 3)
    # return xyz

def ncaco():
    return readpdb(pdbfname()).ncaco()
    # pdb = pdb.subset(het=False, atomnames=['N', 'CA', 'C', 'O'], chains=['A'])
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T.reshape(-1, 4, 3)
    # return xyz

with contextlib.suppress(ImportError):

    from biotite.structure import AtomArray

    def small_atoms():
        """
         Create a simple protein-like atom array for testing.

        Structure contains:
         - 3 residues (ALA, PHE, GLY)
         - Multiple atom types (backbone + sidechain)
         - 2 chains (A, B)
        """
        atom_array = AtomArray(20)

        # Set coordinates (not important for selection tests)
        atom_array.coord = np.zeros((20, 3))

        # Chain A, Residue 1 (ALA)
        for i, (atom_name, element) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")]):
            atom_array.atom_name[i] = atom_name
            atom_array.element[i] = element
            atom_array.res_id[i] = 1
            atom_array.res_name[i] = "ALA"
            atom_array.chain_id[i] = "A"

        # Chain A, Residue 2 (PHE)
        for i, (atom_name, element) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
                                                  ("CG", "C"), ("CD1", "C"), ("CD2", "C")]):
            idx = i + 5
            atom_array.atom_name[idx] = atom_name
            atom_array.element[idx] = element
            atom_array.res_id[idx] = 2
            atom_array.res_name[idx] = "PHE"
            atom_array.chain_id[idx] = "A"

        # Chain B, Residue 3 (GLY)
        for i, (atom_name, element) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]):
            idx = i + 13
            atom_array.atom_name[idx] = atom_name

            atom_array.element[idx] = element
            atom_array.res_id[idx] = 3
            atom_array.res_name[idx] = "GLY"
            atom_array.chain_id[idx] = "B"

        # Add some extra properties
        atom_array.b_factor = np.linspace(0, 100, 20)
        atom_array.occupancy = np.ones(20)
        atom_array.altloc_id = np.full(20, "", dtype="U1")

        # Set first altloc as "A" for testing
        atom_array.altloc_id[0] = "A"

        return atom_array

    def mixed_atoms():
        """
         Create a mixed structure with protein, nucleic acid, and water.
         """
        atom_array = AtomArray(30)

        # Set coordinates (not important for selection tests)
        atom_array.coord = np.zeros((30, 3))

        # Protein residue
        for i, (atom_name, element) in enumerate([("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C")]):
            atom_array.atom_name[i] = atom_name
            atom_array.element[i] = element
            atom_array.res_id[i] = 1
            atom_array.res_name[i] = "TYR"
            atom_array.chain_id[i] = "A"

        # Nucleic acid residue
        for i, (atom_name, element) in enumerate([("P", "P"), ("O5'", "O"), ("C5'", "C"), ("C4'", "C"), ("N1", "N")]):
            idx = i + 5
            atom_array.atom_name[idx] = atom_name
            atom_array.element[idx] = element
            atom_array.res_id[idx] = 2
            atom_array.res_name[idx] = "A"
            atom_array.chain_id[idx] = "B"

        # Water molecules
        for i in range(20):
            idx = i + 10
            atom_array.atom_name[idx] = "O"
            atom_array.element[idx] = "O"
            atom_array.res_id[idx] = i + 10
            atom_array.res_name[idx] = "HOH"
            atom_array.chain_id[idx] = "W"

        # Add some extra properties
        atom_array.b_factor = np.linspace(0, 100, 30)
        atom_array.occupancy = np.ones(30)

        return atom_array

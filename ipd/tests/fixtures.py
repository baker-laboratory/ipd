import contextlib
import numpy as np
import ipd

np.set_printoptions(suppress=True)

def pdbcontents():
    return ipd.dev.load_package_data("tests/pdb/1pgx.pdb1.gz")

def pdbfname():
    return ipd.dev.package_testdata_path("pdb/1pgx.pdb1.gz")

def pdbfnames():
    return [ipd.dev.package_testdata_path("pdb/1pgx.pdb1.gz"), ipd.dev.package_testdata_path("pdb/1qys.pdb1.gz")]

def three_PDBFiles():
    return [ipd.pdb.readpdb(ipd.dev.package_testdata_path(f"pdb/{code}.pdb1.gz")) for code in ["3asl", "1pgx", "1coi"]]
    # return [ipd.dev.load_package_data(ftests/'pdb/{code}.pdb1.gz.pickle') for code in ['3asl', '1pgx', '1coi']]

def pdbfile():
    return ipd.pdb.readpdb(ipd.dev.package_testdata_path("pdb/3asl.pdb1.gz"))
    # return ipd.dev.load_package_data('tests/pdb/3asl.pdb1.gz.pickle')

def pdb1pgx():
    return ipd.pdb.readpdb(ipd.dev.package_testdata_path("pdb/1pgx.pdb1.gz"))
    # return ipd.dev.load_package_data('tests/pdb/1pgx.pdb1.gz.pickle')

def pdb1coi():
    return ipd.pdb.readpdb(ipd.dev.package_testdata_path("pdb/1coi.pdb1.gz"))
    # return ipd.dev.load_package_data('tests/pdb/1coi.pdb1.gz.pickle')

def pdb1qys():
    return ipd.pdb.readpdb(ipd.dev.package_testdata_path("pdb/1qys.pdb1.gz"))
    # return ipd.dev.load_package_data('tests/pdb/1qys.pdb1.gz.pickle')

def ncac():
    return ipd.pdb.readpdb(pdbfname()).ncac()
    # pdb = pdb.subset(atomnames=['N', 'CA', 'C'], chains=['A'])
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T.reshape(-1, 3, 3)
    # return xyz

def ncaco():
    return ipd.pdb.readpdb(pdbfname()).ncaco()
    # pdb = pdb.subset(het=False, atomnames=['N', 'CA', 'C', 'O'], chains=['A'])
    # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T.reshape(-1, 4, 3)
    # return xyz

_fname_map = dict(top7='1qys')

def atoms(fname):
    fname = _fname_map.get(fname, fname)
    if '.' not in fname: fname = f'{fname}.bcif.gz'
    atoms = ipd.pdb.readatoms(ipd.dev.ipd.dev.package_testdata_path(f'pdb/{fname}'))
    return atoms

with contextlib.suppress(ImportError):

    from biotite.structure import AtomArray

    small_atoms = atoms('tiny.pdb')

    def make_mixed_atoms():
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

    mixed_atoms = make_mixed_atoms()

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

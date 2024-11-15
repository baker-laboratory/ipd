import glob
import gzip
import io
import logging
import os
import time

import numpy as np
import tqdm

import ipd
from ipd.pdb.pdbfile import PDBFile

log = logging.getLogger(__name__)

def readfile(fname, **kw):
    if not os.path.exists(fname):
        raise FileNotFoundError(f"no file {fname}")
    finfo = ipd.dev.fname_extensions(fname)
    if finfo.ext == ".cif":
        return ipd.readcif(fname, **kw)
    elif finfo.ext == ".pdb":
        return ipd.pdb.readpdb(fname, **kw)
    else:
        raise ValueError(f"don't know how to read {fname}")

##@timed
def readpdb(fname_or_buf, indatabase=False, **kw):
    pdbatoms, meta, original_contents = read_pdb_atoms(fname_or_buf)
    df = parse_pdb_atoms(pdbatoms)
    code = pdb_code(fname_or_buf) if indatabase else "none"
    resl = -1.0
    if code != "none":
        resl = metadb = ipd.pdb.pdbmeta.resl[code]
    meta.update(code=code, resl=resl)
    df = concatenate_models(df)
    pdb = PDBFile(df, meta, original_contents, **kw)
    return pdb

def pdb_code(fname):
    if len(fname) > 100:
        return "none"
    fname = os.path.basename(fname)
    if len(fname.split(".")[0]) == 4:
        return fname[:4].upper()
    else:
        return "none"

##@timed
def read_pdb_atoms(fname_or_buf, **kw):
    atomlines, meta = dict(), ipd.dev.Bunch(fname=None, cryst1=None)

    if ipd.dev.is_pdb_fname(fname_or_buf):
        meta.fname = fname_or_buf
        opener = gzip.open if fname_or_buf.endswith(".gz") else open
        with opener(fname_or_buf) as inp:
            contents = str(inp.read()).replace(r"\n", "\n")
            # contents = str(inp.read())
    else:
        contents = fname_or_buf
        if contents.count("ATOM  ") == 0 and contents.count("HETATM") == 0:
            raise ValueError(f"bad pdb: {contents}")
    if contents.startswith("b'"):
        contents = contents[2:]

    modelnum = 0
    atomlines[0] = list()
    for i, line in enumerate(contents.splitlines()):
        if line.startswith(("ATOM", "HETATM")):
            atomlines[modelnum].append(line)
        elif line.startswith("MODEL "):
            modelnum = int(line[6:])
            atomlines[modelnum] = list()
        elif line.startswith("CRYST1 "):
            assert not meta.cryst1
            meta.cryst1 = line.strip()

    # ic(len(atomlines))
    assert atomlines

    return {k: "\n".join(v) for k, v in atomlines.items()}, meta, contents

##@timed
def parse_pdb_atoms(atomstr, **kw):
    import pandas as pd

    assert atomstr

    dt = pdbcoldtypes.copy()
    del dt["het"]
    del dt["ai"]
    cr = pdbcolrange.copy()
    cr[0] = 0, cr[1][1]
    cr[1] = 0, cr[1][1]
    converters = dict(
        het=lambda x: x.startswith("HETATM"),
        # logic below allows entries line 'ATOM 123456'
        ai=lambda x: np.int32(x[4:]) if x.startswith("ATOM") else np.int32(x[6:]),
    )

    mdf = dict()
    for m in atomstr.keys():
        df = pd.read_fwf(
            io.StringIO(atomstr[m]),
            names=pdbcolnames,
            colspecs=cr,
            header=None,
            dtype=dt,
            converters=converters,
            na_filter=False,
        )
        # don't understand why pandas doesn't respect the str dtypes from "dt"
        df.an = df.an.astype("a4")
        df.al = df.al.astype("a1")
        df.ch = df.ch.astype("a1")
        df.rn = df.rn.astype("a3")
        # df.rins = df.rins.astype('a1')
        # df.seg = df.seg.astype('a4')
        df.elem = df.elem.astype("a2")
        # df.charge = df.charge.astype('a2')
        # print(df.dtypesb)
        # print(df.memory_usage())

        notalt = np.logical_or(df.al == b"", df.al == b"A")
        df = df[notalt]
        df.drop("al", axis=1, inplace=True)
        mdf[m] = df
    return mdf

##@timed
def concatenate_models(df):
    import pandas as pd

    assert isinstance(df, dict)
    df = {k: v for k, v in df.items() if len(v)}
    for m, d in df.items():
        d["mdl"] = m
    df = pd.concat(df.values())
    return df

def format_atom(atomi=0, atomn="ATOM", idx=" ", resn="ALA", chain="A", resi=0, insert=" ", x=0, y=0, z=0, occ=1, b=0):
    return _atom_record_format.format(**locals())  # type: ignore

##@timed
def find_pdb_files(files_or_pattern, maxsize=99e99, **kw):
    if isinstance(files_or_pattern, str):
        files_or_pattern = [files_or_pattern]
    candidates = list()
    for f in files_or_pattern:
        if not os.path.exists(f):
            candidates.extend(glob.glob(f))
        else:
            candidates.append(f)
    files = list()
    for f in candidates:
        if os.path.getsize(f) > maxsize:
            continue
        if not os.path.exists(f):
            raise ValueError(f"pdb file {f} does not exist")
        files.append(f)
    return files

def load_pdb(
    fname,
    cache=True,
):
    fname = fname.replace(".pickle", "")
    if cache:
        try:
            pdbfile = ipd.load(fname + ".pickle")
            log.info(f'loaded {fname + ".pickle"}')
        except (FileNotFoundError, EOFError, AttributeError) as e:
            if not isinstance(e, FileNotFoundError):
                log.warning(f"cache failure, loading {fname}")
            else:
                log.info(f"cache failure, loading {fname}")
            pdbfile = readpdb(fname)
            ipd.save(pdbfile, fname + ".pickle")
    else:
        log.info(f"loading {fname}")
        pdbfile = readpdb(fname)
    return pdbfile

def load_pdbs(
    files_or_pattern,
    cache=True,
    skip_errors=False,
    pbar=True,
    maxfiles=9e9,
    **kw,
):
    files = find_pdb_files(files_or_pattern, **kw)
    pdbs = dict()
    for fname in tqdm.tqdm(files) if pbar else files:
        t = time.perf_counter()
        try:
            pdbs[fname] = load_pdb(fname, cache)
            if len(pdbs) >= maxfiles:
                break
        except (FileNotFoundError, ValueError) as e:
            if not skip_errors:
                raise e

    return pdbs

def gen_pdbs(
    files_or_pattern,
    cache=True,
    skip_errors=False,
    pbar=True,
    maxfiles=9e9,
    **kw,
):
    files = find_pdb_files(files_or_pattern, **kw)
    pdbs = dict()
    for fname in tqdm.tqdm(files) if pbar else files:
        t = time.perf_counter()
        try:
            yield fname, load_pdb(fname, cache)
            if len(pdbs) >= maxfiles:
                break
        except (FileNotFoundError, ValueError) as e:
            if not skip_errors:
                raise e

# pdb format
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

pdbcolnames = [
    "het",  # "ATOM  "
    "ai",  # Atom serial number.
    "an",  # Atom name.
    "al",  # Alternate location indicato
    "rn",  # Residue name.
    "ch",  # Chain identifier.
    "ri",  # Residue sequence number.
    # 'rins',  # Code for insertion of resid
    "x",  # Orthogonal coordinates for
    "y",  # Orthogonal coordinates for
    "z",  # Orthogonal coordinates for
    "occ",  # Occupancy.
    "bfac",  # Temperature factor (Default
    # 'seg',  # Segment identifier, left-ju
    "elem",  # Element symbol, right-justi
    # 'charge',  # Charge on the atom.
]

pdbcolrange = [
    (1 - 1, 6),
    (7 - 1, 11),
    (13 - 1, 16),
    (17 - 1, 17),
    (18 - 1, 20),
    (22 - 1, 22),
    (23 - 1, 26),
    # (27 - 1, 27),
    (31 - 1, 38),
    (39 - 1, 46),
    (47 - 1, 54),
    (55 - 1, 60),
    (61 - 1, 66),
    # (73 - 1, 76),
    (77 - 1, 78),
    # (79 - 1, 80),
]

pdbcoldtypes = dict(
    het="b",  # Record name
    ai="i4",  # Integer
    an="a4",  # Atom
    al="a1",  # Character
    rn="a3",  # Residue name
    ch="a1",  # Character
    ri="i4",  # Integer
    # rins='a1',  # AChar
    x="f4",  # Real(8.3)
    y="f4",  # Real(8.3)
    z="f4",  # Real(8.3)
    occ="f4",  # Real(6.2)
    bfac="f4",  # Real(6.2)
    # seg='a4',  # Lstring(4)
    elem="a2",  # Lstring(2)
    # charge='a2',  # LString(2)
)

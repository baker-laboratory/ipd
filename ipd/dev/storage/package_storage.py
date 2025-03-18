import gzip
import json
import lzma
import os
import pickle
import ipd

def joinpaths(a, b):
    """Join two filesystem paths.

    Args:
        a (str): First part of the path.
        b (str): Second part of the path.

    Returns:
        str: Combined path.

    Example:
        >>> joinpaths('/home/user', 'data/file.txt')
        '/home/user/data/file.txt'
    """
    a, b = a.rstrip('/'), b.lstrip('/')
    return os.path.join(a, b)

package_dir = os.path.realpath(joinpaths(os.path.dirname(__file__), "../.."))
data_dir = os.path.realpath(joinpaths(os.path.dirname(__file__), "../../data"))
_compression_extensions = ("gz", "tgz", "xz", "txz")

def package_data_path(fname=None, emptyok=True, datadir=data_dir):
    """Get the full path to a package data file.

    Args:
        fname (str, optional): File name to locate. Defaults to None.
        emptyok (bool, optional): If True, returns the path even if the file doesn't exist. Defaults to True.
        datadir (str, optional): Base data directory. Defaults to `data_dir`.

    Returns:
        str: Full path to the data file or None if not found and emptyok is False.

    Example:
        >>> package_data_path('spacegroup_data.pickle') # doctest:+ELLIPSIS
        '.../ipd/data/spacegroup_data.pickle'
    """
    if not fname:
        return datadir
    elif os.path.exists(joinpaths(datadir, fname)):
        return joinpaths(datadir, fname)
    elif os.path.exists(joinpaths(datadir, fname + ".pickle")):
        return joinpaths(datadir, fname + ".pickle")
    elif os.path.exists(joinpaths(datadir, fname + ".pickle.gz")):
        return joinpaths(datadir, fname + ".pickle.gz")
    elif os.path.exists(joinpaths(datadir, fname + ".pickle.xz")):
        return joinpaths(datadir, fname + ".pickle.xz")
    if emptyok:
        return joinpaths(datadir, fname)
    else:
        return None

def package_testdata_path(fname=None, emptyok=True):
    """Get the full path to a test data file.

    Args:
        fname (str, optional): File name to locate. Defaults to None.
        emptyok (bool, optional): If True, returns the path even if the file doesn't exist. Defaults to True.

    Returns:
        str: Full path to the test data file.

    Example:
        >>> package_testdata_path('pdb/1a2n.bcif.gz') # doctest:+ELLIPSIS
        '.../ipd/data/tests/pdb/1a2n.bcif.gz'
    """
    return package_data_path(fname, emptyok, datadir=joinpaths(data_dir, 'tests'))

def load_package_data(fname):
    """Load data from a package file.

    Args:
        fname (str): File name to load.

    Returns:
        object: Loaded data.

    Example:
        >>> data = load_package_data('spacegroup_data.pickle')
        >>> isinstance(data, dict)
        True
    """
    datapath = package_data_path(fname)
    data = load(datapath)
    return data

def have_package_data(fname):
    """Check if a package data file exists.

    Args:
        fname (str): File name to check.

    Returns:
        bool: True if the file exists, False otherwise.

    Example:
        >>> have_package_data('spacegroup_data.pickle')
        True
    """
    datapath = package_data_path(fname, emptyok=False)
    return datapath is not None

def open_package_data(fname):
    """Open a package data file.

    Args:
        fname (str): File name to open.

    Returns:
        file object: Open file object.

    Example:
        >>> with open_package_data('spacegroup_data.pickle') as f:
        ...     data = pickle.load(f)
        >>> isinstance(data, dict)
        True
    """
    if fname.endswith(".xz"):
        return open_lzma_cached(package_data_path(fname))
    else:
        return open(package_data_path(fname), "rb")

def load_json(f):
    """Load JSON data from a file.

    Args:
        f (str): File path.

    Returns:
        dict: Parsed JSON data.

    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        ...     charswritten = f.write('{"key": "value"}')
        >>> load_json(f.name)
        {'key': 'value'}
    """
    with open(f, "r") as inp:
        return json.load(inp)

def dump_json(j, f, indent=True):
    """Dump JSON data to a file.

    Args:
        j (dict): JSON data to write.
        f (str): File path.
        indent (bool, optional): If True, formats JSON with indentation. Defaults to True.

    Example:
        >>> import tempfile
        >>> data = {"key": "value"}
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        ...     dump_json(data, f.name)
        >>> load_json(f.name)
        {'key': 'value'}
    """
    with open(f, "w") as out:
        return json.dump(j, out, indent=indent)

def is_pickle_fname(fname):
    """Check if a file name refers to a pickle file.

    Args:
        fname (str): File name.

    Returns:
        bool: True if the file is a pickle file, False otherwise.

    Example:
        >>> is_pickle_fname('data.pickle')
        True
        >>> is_pickle_fname('data.json')
        False
    """
    return os.path.basename(fname).count(".pickle") > 0

class open_lzma_cached:
    """Context manager for opening LZMA-compressed files with caching.

    Args:
        fname (str): File name.
        mode (str, optional): File mode. Defaults to "rb".
    """

    def __init__(self, fname, mode="rb"):
        self.file_obj = lzma.open(fname, mode)

    def __enter__(self):
        return self.file_obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.file_obj.close()

def is_pdb_fname(fn, maxlen=1000):
    """Check if a file name is a PDB file.

    Args:
        fn (str): File name.
        maxlen (int, optional): Maximum allowed length. Defaults to 1000.

    Returns:
        bool: True if the file is a PDB file, False otherwise.

    Raises:
        ValueError: If the file format is unknown.
    """
    if len(fn) > maxlen:
        return False
    elif len(fn.split()) > 1:
        return False
    elif not os.path.exists(fn):
        return False
    elif fn.endswith((".pdb.gz", ".pdb")):
        return True
    elif fn[-4:-1] == "pdb" and fn[-1].isnumeric():
        return True
    elif fn.endswith(".gz") and fn[-8:-4] == ".pdb" and fn[-4].isnumeric():
        return True
    else:
        raise ValueError(f"Can't tell if is PDB fname: {fn}")

def package_testcif_path(pdbcode=None, emptyok=False):
    if not pdbcode: return joinpaths(package_data_path(), 'pdb')
    return package_testdata_path(f'pdb/{pdbcode}.bcif.gz')

def open_package_file(fname):
    return open(joinpaths(package_dir, fname))

def save_package_data(stuff, fname):
    return save(stuff, package_data_path(fname))

def decompress_lzma_file(fn, overwrite=True, use_existing=False, missing_ok=False):
    assert fn.endswith(".xz") and not fn.endswith(".xz.xz")
    if missing_ok and not os.path.exists(fn):
        return
    assert os.path.exists(fn)
    exists = os.path.exists(fn[-3:])
    if exists and not overwrite and not use_existing:
        assert not exists, "cant overwrite: " + fn[:-3]
    if not exists or (exists and overwrite):
        with lzma.open(fn, "rb") as inp:
            with open(fn[:-3], "wb") as out:
                out.write(inp.read())

def fname_extensions(fname):
    """Split fname dir/base.ext.compression into parts."""
    d, b = os.path.split(fname)
    s = b.split(".")
    if len(s) == 1:
        d, f, e, c = d, b, "", ""
    elif len(s) == 2:
        if s[1] in _compression_extensions:
            d, f, e, c = d, s[0], "", s[1]
        else:
            d, f, e, c = d, s[0], s[1], ""
    elif len(s) == 3:
        ext1, ext2 = s[-2:]
        if ext2 in _compression_extensions:
            if ext1 == "tar":
                d, f, e, c = d, ".".join(s[:-2]), "", "tar." + ext2
            else:
                d, f, e, c = d, ".".join(s[:-2]), ext1, ext2
        else:
            d, f, e, c = d, ".".join(s[:-1]), ext2, ""
    elif len(s) > 3:
        ext1, ext2, ext3 = s[-3:]
        if ext2 == "tar":
            d, f, e, c = d, ".".join(s[:-3]), ext1, "tar." + ext3
        elif ext3 in _compression_extensions:
            d, f, e, c = d, ".".join(s[:-2]), ext2, ext3
        else:
            d, f, e, c = d, ".".join(s[:-1]), ext3, ""
    else:
        assert 0

    # ic(e)
    # ic(f'{b}.{e}' if e else b)
    directory = f"{d}/" if d else ""
    base = f  # type: ignore
    ext = f".{e}" if e else ""  # type: ignore
    compression = f".{c}" if c else ""  # type: ignore
    basename = b
    baseext = f"{f}.{e}" if e else f  # type: ignore
    extcomp = ""
    if e and c:  # type: ignore
        extcomp = f".{e}.{c}"
    elif e:  # type: ignore
        extcomp = f".{e}"
    elif c:  # type: ignore
        extcomp = f".{c}"
    uncomp = f"{directory}{baseext}"

    return ipd.dev.Bunch(
        directory=directory,
        base=base,
        ext=ext,
        compression=compression,
        basename=basename,
        baseext=baseext,
        extcomp=extcomp,
        uncomp=uncomp,
    )

def load(fname, **kw):
    import numpy as np
    if fname.count(".") == 0 or is_pickle_fname(fname):
        return load_pickle(fname, **kw)
    elif fname.endswith(".nc"):
        import xarray  # type: ignore
        return xarray.load_dataset(fname, **kw)
    elif fname.endswith(".npy"):
        return np.load(fname, **kw)
    elif fname.endswith(".json"):
        with open(fname) as inp:
            return json.load(inp)
    elif fname.endswith(".json.xz"):
        with lzma.open(fname, "rb") as inp:
            return json.load(inp)
    elif fname.endswith(".gz") and fname[-8:-4] == ".pdb" and fname[-4].isnumeric():
        with gzip.open(fname, "rb") as inp:
            # kinda confused why this \n replacement is needed....
            return str(inp.read()).replace(r"\n", "\n")
    elif fname.endswith(".xz"):
        with open_lzma_cached(fname, **kw) as inp:
            return inp.read()
    else:
        raise ValueError("dont know how to handle file " + fname)

def load_pickle(fname, add_dotpickle=True, assume_lzma=False, **kw):
    opener = open
    if fname.endswith(".xz"):
        opener = open_lzma_cached
    elif fname.endswith(".gz"):
        opener = gzip.open
    elif not fname.endswith(".pickle"):
        if assume_lzma:
            opener = open_lzma_cached
            fname += ".pickle.xz"
        else:
            fname += ".pickle"
    # print(f'load_pickle {fname} {opener}')
    with opener(fname, "rb") as inp:
        stuff = pickle.load(inp)  # type: ignore
        if isinstance(stuff, dict):
            if "__I_WAS_A_BUNCH_AND_THIS_IS_MY_SPECIAL_STUFF__" in stuff:
                _special = stuff["__I_WAS_A_BUNCH_AND_THIS_IS_MY_SPECIAL_STUFF__"]
                del stuff["__I_WAS_A_BUNCH_AND_THIS_IS_MY_SPECIAL_STUFF__"]
                stuff = ipd.dev.Bunch(stuff)
                stuff._special = _special

    return stuff

def save(stuff, fname, **kw):
    finfo = fname_extensions(fname)
    if finfo.directory:
        os.makedirs(finfo.directory, exist_ok=True)
    if finfo.ext in (".pdb", ".cif"):
        ipd.pdb.dumpstruct(fname, stuff, **kw)
    elif finfo.ext == ".nc":
        import xarray  # type: ignore
        if not isinstance(stuff, xarray.Dataset):
            raise ValueError("can only save xarray.Dataset as .nc file")
        stuff.to_netcdf(fname)
    elif finfo.ext == ".npy":
        import numpy as np
        np.save(fname, stuff)
    elif fname.count(".") == 0 or is_pickle_fname(fname):
        save_pickle(stuff, fname, **kw)
    elif fname.endswith(".json"):
        with open(fname, "w") as out:
            out.write(json.dumps(stuff, sort_keys=True, indent=4))
    elif fname.endswith(".json.xz"):
        jsonstr = json.dumps(stuff, sort_keys=True, indent=4)
        with lzma.open(fname, "wb") as out:
            out.write(jsonstr.encode())
    else:
        raise ValueError("dont know now to handle file " + fname)

def save_pickle(stuff, fname, add_dotpickle=True, uselzma=False, **kw):
    opener = open
    if isinstance(stuff, ipd.dev.Bunch):
        # pickle as dict to avoid version problems or whatever
        _special = stuff._special
        stuff = dict(stuff)
        stuff["__I_WAS_A_BUNCH_AND_THIS_IS_MY_SPECIAL_STUFF__"] = _special
    if fname.endswith(".xz"):
        assert fname.endswith(".pickle.xz")
        cachefile = os.path.expanduser(os.path.relpath(f"~/.cache/ipd/{fname}"))
        opener = lzma.open
        if os.path.exists(cachefile):
            os.remove(cachefile)
    elif fname.endswith(".gz"):
        assert fname.endswith(".pickle.gz")
        opener = gzip.open
    elif uselzma:
        opener = lzma.open
        if not fname.endswith(".pickle"):
            fname += ".pickle"
        fname += ".xz"
    if not os.path.basename(fname).count("."):
        fname += ".pickle"
    with opener(fname, "wb") as out:
        pickle.dump(stuff, out)  # type: ignore

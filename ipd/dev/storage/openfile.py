import gzip
from pathlib import Path
from ipd.dev.decorators import iterize_on_first_param

openers = {'.gz': gzip.open, '': open}

def compressed_open(fname, **kw):
    for ext, opener in openers.items():
        if fname.endswith(ext):
            return ext, opener(fname, **kw)
    assert 0, 'should not be possible'

@iterize_on_first_param(basetype=(str, Path))
def decompressed_fname(fname):
    for ext in openers:
        if ext and fname.endswith(ext):
            return fname[:-len(ext)]
    return fname

@iterize_on_first_param(basetype=(str, Path))
def openfile(fname, mode='r', **kw):
    """
    open a possibly compressed file based on suffix
    """
    compression, file = compressed_open(fname, mode=mode)
    return file

@iterize_on_first_param(basetype=(str, Path))
def readfile(fname, **kw):
    """
    read a possibly compressed file, inferring format txt, json, yaml, pdb, cif, etc
    """
    file = openfile(fname, **kw)
    # raise NotImplementedError
    val = file.read()
    file.close()
    return val

def closefiles(files):
    if isinstance(files, list):
        for f in files:
            if isinstance(f, list): [g.close() for g in f]
            else: f.close()
    else:
        files.close()

def istextfile(file):
    if not file: return False
    if isinstance(file.mode, int): return False
    if 't' in file.mode: return True
    if 'b' not in file.mode: return True
    return False

def isbinfile(file):
    if not file: return False
    return not istextfile(file)

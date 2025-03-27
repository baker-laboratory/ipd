import contextlib
import gzip
import zipfile
import lzma
import tarfile
from pathlib import Path
from ipd.dev.decorators import iterize_on_first_param

@contextlib.contextmanager
def zipopen(zip, **kw):
    try:
        with zipfile.ZipFile(zip, **kw) as myzip:
            with myzip.open('my_file.txt') as myfile:
                yield myfile
    finally:
        pass

openers = {'.gz': gzip.open, '.zip': zipopen, '.xz': lzma.open, '.tar': tarfile.open, '': open}

def compressed_open(filepath, **kw):
    for ext, opener in openers.items():
        if str(filepath).endswith(ext):
            return ext, opener(filepath, **kw)
    assert 0, 'should not be possible'

@iterize_on_first_param(basetype=(str, Path))
def decompressed_fname(filepath):
    filepath, orig = str(filepath), type(filepath)
    for ext in openers:
        if ext and filepath.endswith(ext):
            return decompressed_fname(orig(filepath[:-len(ext)]))
    return orig(filepath)

@iterize_on_first_param(basetype=(str, Path))
def openfile(filepath, mode='r', **kw):
    """
    open a possibly compressed file based on suffix
    """
    compression, file = compressed_open(filepath, mode=mode)
    return file

@iterize_on_first_param(basetype=(str, Path))
def readfile(filepath, **kw):
    """
    read a possibly compressed file, inferring format txt, json, yaml, pdb, cif, etc
    """
    file = openfile(filepath, **kw)
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

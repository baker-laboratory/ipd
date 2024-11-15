import functools
import glob
import re

import numpy as np
from deferred_import import deferred_import  # type: ignore
import ipd

pyrosetta = deferred_import("pyrosetta")

def get_rosetta_symfile_path(name):
    name = name.upper().replace("M_", "m_")
    if name.endswith(".sym"):
        name = name[:-4]
    name = ipd.dev.package_data_path(f"rosetta_symdef/{name}")
    pattern = name + "*.sym"  # type: ignore
    ic(pattern)  # type: ignore
    g = glob.glob(pattern)
    assert len(g) == 1
    return g[0]

@functools.lru_cache()
def get_rosetta_symfile_contents(name):
    path = get_rosetta_symfile_path(name)
    print(f"reading symdef {path}")
    with ipd.open_package_data(path) as inp:
        return inp.read()

@functools.lru_cache()
def get_rosetta_symdata(name):
    if name is None:
        return None
    ss = pyrosetta.rosetta.std.stringstream(get_rosetta_symfile_contents(name))
    d = pyrosetta.rosetta.core.conformation.symmetry.SymmData()
    d.read_symmetry_data_from_stream(ss)
    return d

def get_rosetta_symdata_modified(name, string_substitutions=None, scale_positions=None):
    rosetta_symfilestr = get_rosetta_symfile_contents(name)
    if scale_positions is not None:
        if string_substitutions is None:
            string_substitutions = dict()
        for line in rosetta_symfilestr.splitlines():
            if not line.startswith("xyz"):
                continue
            if isinstance(scale_positions, np.ndarray):
                for posstr in re.split(r"\s+", line)[-3:]:
                    tmp = np.array([float(x) for x in posstr.split(",")])
                    x, y, z = tmp * scale_positions
                    string_substitutions[posstr] = "%f,%f,%f" % (x, y, z)
            else:
                posstr = re.split(r"\s+", line)[-1]
                x, y, z = [float(x) * scale_positions for x in posstr.split(",")]
                string_substitutions[posstr] = "%f,%f,%f" % (x, y, z)
    if string_substitutions is not None:
        for k, v in string_substitutions.items():
            rosetta_symfilestr = rosetta_symfilestr.replace(k, v)
    ss = pyrosetta.rosetta.std.stringstream(rosetta_symfilestr)
    d = pyrosetta.rosetta.core.conformation.symmetry.SymmData()
    d.read_symmetry_data_from_stream(ss)
    return d, rosetta_symfilestr

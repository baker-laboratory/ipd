from pyrosetta import Pose  # type: ignore
import ipd

def is_rosetta_pose(toshow):
    return isinstance(toshow, Pose)

def pymol_load_pose(pose, name):
    tmpdir = tempfile.mkdtemp()  # type: ignore
    fname = tmpdir + "/" + name + ".pdb"
    pose.dump_pdb(fname)
    pymol.cmd.load(fname)  # type: ignore

@ipd.viz.pymol_load.register(Pose)
def _(toshow, name=None, state=None, **kw):
    name = name or "rif_thing"
    state["seenit"][name] += 1  # type: ignore
    name += "_%i" % state["seenit"][name]  # type: ignore
    pymol_load_pose(toshow, name, **kw)
    state["last_obj"] = name  # type: ignore
    return state

import willutil as wu
from pyrosetta import Pose


def is_rosetta_pose(toshow):
    return isinstance(toshow, Pose)


def pymol_load_pose(pose, name):
    tmpdir = tempfile.mkdtemp()
    fname = tmpdir + "/" + name + ".pdb"
    pose.dump_pdb(fname)
    pymol.cmd.load(fname)


@wu.viz.pymol_load.register(Pose)
def _(toshow, name=None, state=None, **kw):
    name = name or "rif_thing"
    state["seenit"][name] += 1
    name += "_%i" % state["seenit"][name]
    pymol_load_pose(toshow, name, **kw)
    state["last_obj"] = name
    return state

import random
import sys
import tempfile

import ipd

try:
    import pymol  # type: ignore
except ImportError:
    pass

if 'pymol' in sys.modules:

    @ipd.viz.pymol_scene
    @ipd.viz.pymol_load.register(ipd.voxel.Voxel)
    def pymol_load_Voxel(vox, name='Voxel', sym=None, **kw):
        ipd.viz.show_ndarray_point_or_vec(vox.xyz - vox.lb, name=name + '_xyz', sphere=1, **kw)
        with tempfile.TemporaryDirectory() as d:
            vox.dump_ccp4(f'{d}/{name}_map.ccp4')
            pymol.cmd.load(f'{d}/{name}_map.ccp4')  # type: ignore
            pymol.cmd.isomesh(f'{name}_iso_{str(random.random())[2:]}', f'{name}_map')  # type: ignore
            pymol.cmd.delete(f'{d}/{name}_map')  # type: ignore

    @ipd.viz.pymol_scene
    @ipd.viz.pymol_load.register(ipd.voxel.VoxRB)
    def pymol_load_VoxRB(rb, name='Voxel', sym=None, **kw):
        kw = ipd.dev.Bunch(kw)
        kw.set_if_missing('sphere', 1)
        xyz = ipd.h.xform(rb._vizpos.cpu(), rb.xyz.cpu())
        if xyz.ndim == 2: xyz = xyz[None]
        ipd.viz.show_ndarray_point_or_vec(xyz, name=name + '_xyz', **kw)

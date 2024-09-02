import random
import sys
import willutil as wu
import ipd
import gemmi
import torch as th
import numpy as np
import tempfile
try:
    import pymol
except ImportError:
    pass

if 'pymol' in sys.modules:

    @wu.viz.pymol_frame
    @wu.viz.pymol_load.register(ipd.voxel.Voxel)
    def pymol_load_Voxel(vox, name='Voxel', sym=None, **kw):
        wu.viz.show_ndarray_point_or_vec(vox.xyz-vox.lb, name=name + '_xyz', sphere=1, **kw)
        with tempfile.TemporaryDirectory() as d:
            vox.dump_ccp4(f'{d}/{name}_map.ccp4')
            pymol.cmd.load(f'{d}/{name}_map.ccp4')
            pymol.cmd.isomesh(f'{name}_iso_{str(random.random())[2:]}', f'{name}_map')
            pymol.cmd.delete(f'{d}/{name}_map')

    @wu.viz.pymol_frame
    @wu.viz.pymol_load.register(ipd.voxel.VoxRB)
    def pymol_load_VoxRB(rb, name='Voxel', sym=None, **kw):
        kw = wu.Bunch(kw)
        kw.set_if_missing('sphere', 1)
        xyz = wu.h.xform(rb._vizpos.cpu(), rb.xyz.cpu())
        if xyz.ndim ==2: xyz = xyz[None]
        wu.viz.show_ndarray_point_or_vec(xyz, name=name + '_xyz', **kw)



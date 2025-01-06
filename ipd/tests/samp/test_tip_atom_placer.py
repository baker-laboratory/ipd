import pytest

from ipd.lazy_import import lazyimport

th = lazyimport('torch')

import ipd
from ipd import h

pytest.skip(allow_module_level=True)

def main():
    test_place_tip_atoms()
    test_tgt_frames()
    test_tip_frames()
    test_hypercone_samp()
    test_tip_atom_definitions()
    test_tip_atom_target()
    test_tip_atom_groups()

@pytest.mark.fast
def test_place_tip_atoms():
    tgt = ipd.samp.TipAtomTarget.from_pdb(ipd.tests.path('pdb/dna_example.pdb'), clashthresh=2.0)
    tips = ipd.samp.get_tip_atom_groups()
    tgt.place_tip_atoms(tips)

def helper_test_ray_frames(thing, inv=False):
    for frame, rays in zip(thing.donacc_frames(), [thing.don, thing.acc]):
        ori = rays[:, :, 1]
        cen = rays[:, :, 0] + 2.7*ori
        if len(frame) and inv:
            assert th.allclose(cen, h.xform(frame, h.point([0, 0, 0])), atol=1e-3)
            assert th.allclose(ori, h.xform(frame, h.vec([0, 0, 1])), atol=1e-3)
        if len(frame) and not inv:
            assert th.allclose(h.xform(frame, cen), h.point([0, 0, 0]), atol=1e-3)
            assert th.allclose(h.xform(frame, ori), h.vec([0, 0, 1]), atol=1e-3)

@pytest.mark.fast
def test_tip_frames():
    for tip in ipd.samp.get_tip_atom_groups():
        helper_test_ray_frames(tip)

@pytest.mark.fast
def test_tgt_frames():
    fname = ipd.tests.path('pdb/dna_example.pdb')
    tgt = ipd.samp.TipAtomTarget.from_pdb(fname, clashthresh=2.0)
    helper_test_ray_frames(tgt, inv=True)

@pytest.mark.fast
def test_hypercone_samp():
    for _ in range(3):
        spacing = th.rand(1).item() * 30 * th.pi / 180
        # spacing = 15 * th.pi / 180
        longtol = 2 * th.pi
        shorttol = (1 + th.rand(1).item() * 2) * spacing
        # shorttol = spacing*2
        ntest = 100
        xhtor = ipd.samp.quat_torus_xform(resl=spacing, maxtip=shorttol, ringang=longtol)
        xsamp = ipd.samp.randxform(ntest, orimax=spacing, cartmax=0)
        xsamp = h.rot([0, 0, 1], th.rand(ntest) * th.pi, device='cuda') @ xsamp
        xrel = h.inv(xhtor)[None] @ xsamp[:, None]
        ang = h.angle(xrel).min(1).values
        if th.sum(ang > spacing / 2) / len(ang) > 0.1:
            print(f'FAIL {shorttol/spacing:7.3f} {spacing/ang.max()}')
        del xsamp, xrel, ang
    # ic(h.angle(x).max()*180/th.pi)
    # ic(h.angle(x[idx]).max()*180/th.pi)
    # ic(x.shape)
    # ipd.showme(x, xyzlen=(3, 3, 3), showneg=0)
    # ipd.showme(x[idx], xyzlen=(3, 3, 3), showneg=0)

    #  double spacing = sqrt(3.0)*reslang;
    #  int const n1 = std::ceil(long_tol/spacing);
    #  spacing = long_tol/n1;
    #  int const n2 = std::ceil(short_tol/spacing);
    #  for(int i =  0 ; i <  n1; ++i){
    #  for(int j = -n2; j <= n2; ++j){
    #  for(int k = -n2; k <= n2; ++k){
    #  for(int o =  0;  o <=  1; ++o){
    #      // if(o && ( j==n2 || k==n2 ) ) continue;
    #      double const wx = i*spacing + (o?spacing/2.0:0.0) - long_tol/2.0;
    #      double const  w = cos( wx/2.0 );
    #      double const  x = sin( wx/2.0 );
    #      double const  y = ( j*spacing + (o?spacing/2.0:0.0) ) / 2.0;
    #      double const  z = ( k*spacing + (o?spacing/2.0:0.0) ) / 2.0;
    #      if( y*y + z*z > short_tol*short_tol/4.0 ) continue;
    #      Quaterniond q(w,x,y,z);
    #      q.normalize();

@pytest.mark.fast
def test_tip_atom_groups():
    ipd.samp.get_tip_atom_groups()

@pytest.mark.fast
def test_tip_atom_definitions():
    ipd.samp.get_tip_atom_groups()
    # print()
    # assert 0

@pytest.mark.fast
def test_tip_atom_target():
    fname = ipd.tests.path('pdb/dna_example.pdb')
    tgtres = None
    # tgtres = [8]
    tgt = ipd.samp.TipAtomTarget.from_pdb(fname, tgtres, clashthresh=2.0)

    # ipd.save((tgt.don, tgt.acc), '/home/sheffler/project/neorifgen/dna_example.pickle')
    refdon, refacc = ipd.tests.load('ref/dna_example')
    # ipd.showme(tgt)
    # ipd.showme(tgt.vox.xyz, sphere=0.3)
    # import pymol
    # pymol.cmd.load(fname)
    # pymol.cmd.show('line')

    assert th.allclose(refdon, tgt.don.cpu())
    assert th.allclose(refacc, tgt.acc.cpu())

if __name__ == '__main__':
    main()

import tempfile

import numpy as np

import ipd

def main():
    from ipd.tests import fixtures as f

    _test_cb_fillin()
    test_pdbfile_coords(f.pdb1pgx())  # type: ignore
    test_pdbfile_dump(f.pdb1pgx())
    test_pdb_bbcoords_sym(f.pdb1coi())
    test_pdb_masks(f.pdb1pgx())
    test_pdb_xyz(f.pdb1pgx())
    test_pdb_renumber(f.pdb1pgx())
    test_pdb_multimodel(f.pdb1coi())
    test_pdb_mask(f.pdb1pgx())
    test_pdb_bbcoords(f.pdb1pgx())
    test_pdbfile(f.pdbfile())

def _test_cb_fillin():
    # fname = '/home/sheffler/rpxdock/rpxdock_master/rpxdock/data/pdb/C3_1na0-1_1.pdb.gz'
    fname = '/home/sheffler/rpxdock/rpxdock_master/rpxdock/data/pdb/C2_3hm4_1.pdb'
    pdb = ipd.pdb.readpdb(fname)
    ncaco, _mask = pdb.atomcoords(["n", "ca", "c", "o"], nomask=True)

    print(ncaco.shape)

def test_pdbfile_dump(pdb1pgx):
    pdb1pgx.dump_pdb("test.pdb")
    with tempfile.TemporaryDirectory() as td:
        pdb1pgx.dump_pdb(f"{td}/tmp.pdb")
        pdb2 = ipd.pdb.readpdb(f"{td}/tmp.pdb")
        assert np.allclose(pdb2.df.het, pdb1pgx.df.het)
        assert np.allclose(pdb2.df.ai, pdb1pgx.df.ai)
        assert np.all(pdb2.df.an == pdb1pgx.df.an)
        assert np.allclose(pdb2.df.ri, pdb1pgx.df.ri)
        assert np.all(pdb2.df.rn == pdb1pgx.df.rn)
        assert np.allclose(pdb2.df.x, pdb1pgx.df.x)
        assert np.allclose(pdb2.df.y, pdb1pgx.df.y)
        assert np.allclose(pdb2.df.z, pdb1pgx.df.z)
        assert np.allclose(pdb2.df.bfac, pdb1pgx.df.bfac)
        assert np.allclose(pdb2.df.occ, pdb1pgx.df.occ)

def test_pdb_masks(pdb1pgx):
    pdb = pdb1pgx.copy()
    assert np.allclose(pdb.aamask, pdb.camask())
    assert np.allclose(pdb.cbmask(), pdb.atommask("CB"))
    anames = "n ca c o cb".split()
    mask = pdb.atommask(anames)
    assert len(mask) == 70
    for i, an in enumerate(anames):
        assert np.allclose(pdb.atommask(an), mask[:, i])
        assert np.sum(mask[:, i]) == 66 if i == 4 else 70

    pdb2 = pdb.subset(removeatoms=[20, 41])  # THR 10 C and VAL 13 CB
    assert len(pdb.df) - 2 == len(pdb2.df)
    mask2 = pdb2.atommask(anames)
    assert mask.shape == mask2.shape
    assert np.sum(pdb.an == b"C") == np.sum(pdb2.an == b"C") + 1
    assert np.sum(pdb.an == b"CB") == np.sum(pdb2.an == b"CB") + 1

    assert np.sum(mask != mask2) == 2
    assert np.sum(mask[2] != mask2[2]) == 1
    assert np.sum(mask[:, 4] != mask2[:, 4]) == 1

    rstart = pdb.ri[0]
    assert mask[10 - rstart, 2] and not mask2[2, 2]
    assert mask[13 - rstart, 4] and not mask2[5, 4]

def test_pdb_xyz(pdb1pgx):
    p = pdb1pgx.subset(het=False).renumber_from_0()
    assert np.allclose(p.xyz(7, 1), p.xyz(7, "CA"))
    assert np.allclose(p.xyz(29, 1), p.xyz(29, "CA"))
    assert np.allclose(p.xyz(7, 4), p.xyz(7, "CB"))
    assert np.allclose(p.xyz(29, 4), p.xyz(29, "CB"))
    assert np.allclose(p.xyz(7, 6), p.xyz(7, "CG2"))

def test_pdb_multimodel(pdb1coi):
    bb = pdb1coi.bb()
    bb0 = pdb1coi.subset(modelidx=0).bb()
    bb1 = pdb1coi.subset(modelidx=1).bb()
    bb2 = pdb1coi.subset(modelidx=2).bb()
    assert bb.shape == (87, 5, 3)
    assert bb0.shape == (29, 5, 3)
    assert bb1.shape == (29, 5, 3)
    assert bb2.shape == (29, 5, 3)

def test_pdb_renumber(pdb1pgx):
    p = pdb1pgx
    assert p.ri[0] == 8
    p.renumber_from_0()
    assert p.ri[0] == 0

def test_pdb_mask(pdb1pgx):
    pdb = pdb1pgx
    # print(pdbfile.df)
    # print(pdbfile.seq)
    # print(pdbfile.code)
    camask = pdb.camask()
    cbmask = pdb.cbmask(aaonly=False)
    assert np.all(np.logical_and(cbmask, camask) == cbmask)
    nca = np.sum(pdb.df.an == b"CA")
    ncb = np.sum(pdb.df.an == b"CB")
    assert np.sum(pdb.camask()) == nca
    assert np.sum(pdb.cbmask()) == ncb
    ngly = np.sum((pdb.df.rn == b"GLY") * (pdb.df.an == b"CA"))
    assert nca - ncb == ngly
    assert nca - np.sum(pdb.cbmask()) == ngly
    p = pdb.subset(het=False)
    assert p.sequence() == pdb.sequence().replace("Z", "")

    seq = p.sequence()
    cbmask = pdb.cbmask(aaonly=True)
    # ic(len(seq), sum(cbmask))
    assert len(seq) == np.sum(camask)
    for s, m in zip(pdb.seq, cbmask):
        assert m == (s != "G")
    # isgly = np.array(list(seq)) == 'G'
    # wgly = np.where(isgly)[0]
    # ic(wgly)
    # ic(cbmask[wgly])

def test_pdb_bbcoords(pdb1pgx):
    pdb = pdb1pgx
    bb = pdb.bb()
    assert np.all(2 > ipd.homog.hnorm(bb[:, 0] - bb[:, 1]))
    assert np.all(2 > ipd.homog.hnorm(bb[:, 1] - bb[:, 2]))
    assert np.all(2 > ipd.homog.hnorm(bb[:, 2] - bb[:, 3]))
    cbdist = ipd.homog.hnorm(bb[:, 1] - bb[:, 4])
    mask = pdb.cbmask()
    hascb = bb[:, 4, 0] < 9e8
    assert np.all(hascb == mask)
    assert np.all(2 > cbdist[hascb])

def test_pdb_bbcoords_sym(pdb1coi):
    pdb = pdb1coi.subset(het=False)
    assert pdb.bb().shape == (87, 5, 3)
    pdb.assign_chains_sym()
    bbs = pdb.bb(splitchains=True)
    # wu.showme(pdb)
    assert len(bbs) == 3
    assert bbs[0].shape == (29, 5, 3)

def test_pdb_bbcoords2(pdb1pgx):
    ncaco = pdb1pgx.ncaco()
    cb0 = pdb1pgx.subset(atomnames=["CB"])
    camask = pdb1pgx.camask()
    cbmask = pdb1pgx.cbmask(aaonly=True)
    assert np.sum(camask) == len(cbmask)
    seq = pdb1pgx.sequence()
    # ic(len(seq), len(cbmask), len(camask))
    # ic(seq)
    assert len(seq) == len(cbmask)
    wcb = np.where(cbmask)[0]
    cb = 9e9 * np.ones((len(ncaco), 3))
    cb[wcb, 0] = cb0.df.x
    cb[wcb, 1] = cb0.df.y
    cb[wcb, 2] = cb0.df.z
    xyz = np.concatenate([ncaco, cb[:, None]], axis=1)

    assert np.allclose(xyz, pdb1pgx.bb())

def test_pdbfile(pdbfile):
    # print(pdbfile.df)
    # ic(pdbfile.nreshet)
    assert pdbfile.nreshet == 85
    a = pdbfile.subset("A")
    b = pdbfile.subset("B")
    assert a.nres + b.nres == pdbfile.nres
    assert np.all(a.df.ch == b"A")
    assert np.all(b.df.ch == b"B")

if __name__ == "__main__":
    main()

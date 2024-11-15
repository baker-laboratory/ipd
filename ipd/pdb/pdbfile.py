import collections
import os
from difflib import SequenceMatcher

import numpy as np

import ipd

_default_tol = ipd.dev.Bunch(rms=2.0, translation=1.0, angle=np.radians(5.0), seqmatch=0.8)

class PDBFile:
    # @timed
    def __init__(
        self,
        df,
        meta,
        original_contents,
        renumber_by_model=True,
        renumber_from_0=False,
        removehet=False,
        **kw,
    ):
        self.init(df, meta, original_contents, renumber_by_model, renumber_from_0, removehet, **kw)

    # @timed
    def init(
        self,
        df,
        meta,
        original_contents,
        renumber_by_model=False,
        renumber_from_0=False,
        removehet=False,
        **kw,
    ):
        self.original_contents = original_contents
        self.meta = meta.copy()
        self.code = meta.code
        self.resl = meta.resl
        self.cryst1, self.spacegroup, self.cellgeom = meta.cryst1, None, None
        if self.cryst1:
            # cut off at 66 chars, avoid end of line nsub or whatever it is
            s = self.cryst1[:66].split()
            self.cellgeom = np.array([float(x) for x in s[1:7]])
            self.spacegroup = " ".join(s[7:])
        # ic(self.cryst1)
        df = df.copy()
        df.reset_index(inplace=True, drop=True)
        self.df = df
        if renumber_by_model:
            self.renumber_by_model()
        if renumber_from_0:
            self.renumber_from_0()
        if removehet:
            self.removehet()
        self.chainseq = _atomrecords_to_chainseq(df)
        self.seqhet = str.join("", self.chainseq.values())
        self.seq = self.seqhet.replace("Z", "")
        # ic(self.seq)
        self.nres = len(self.seq)
        self.nreshet = len(self.seqhet)
        self.nchain = len(self.chainseq)
        self.fname = meta.fname
        try:
            self.aamask = self.atommask("CA", aaonly=False, quiet=True)
        except:  # noqa
            pass
        self.mmcif_info = dict()

    def copy(self, **kw):
        return PDBFile(self.df, self.meta, self.original_contents, **kw)

    def set_cif_info(self, cifdict, **kw):
        self.mmcif_info.update(cifdict)

    def __getattr__(self, k):
        "allow dot access for fields of self.df from self"
        if k == "df":
            raise AttributeError
        elif k in self.df.columns:
            return getattr(self.df, k)
        else:
            raise AttributeError(k)

    def renumber_by_model(self):
        ri_per_model = np.max(self.df.ri) - np.min(self.df.ri) + 1
        ai_per_model = np.max(self.df.ai) - np.min(self.df.ai) + 1
        for m in self.models():
            i = self.modelidx(m)
            idx = self.df.mdl == m
            self.df.ri += np.where(idx, i * ri_per_model, 0)
            self.df.ai += np.where(idx, i * ai_per_model, 0)
        return self

    def natom(self):
        return len(self.df)

    def getres(self, ri):
        r = self.df[self.df.ri == ri].copy()
        r.reset_index(inplace=True, drop=True)
        return r

    def xyz(self, ir, ia):
        r = self.getres(ir)
        if isinstance(ia, int):
            return r.x[ia], r.y[ia], r.z[ia]
        if isinstance(ia, str):
            ia = ia.encode()
        if isinstance(ia, bytes):
            return float(r.x[r.an == ia].iloc[0]), float(r.y[r.an == ia].iloc[0]), float(r.z[r.an == ia].iloc[0])
        raise ValueError(ia)

    def renumber_from_0(self, unique_chains=True):
        assert np.all(self.het == np.sort(self.het))
        if unique_chains:
            chmap = dict()
            for i, c in enumerate(sorted(set([c for c in ipd.pdb.all_pymol_chains if not c.isnumeric()]))):
                chmap[c.encode()] = i
            chmap[b''] = 0
            maxres = np.max(self.df.ri) + 1
            resdelta = np.array([chmap[_] * maxres for _ in self.df.ch])
            self.df.loc[:, "ri"] = self.df.ri + resdelta
        d = {ri: i for i, ri in enumerate(np.unique(self.ri))}
        self.df["ri"] = [d[ri] for ri in self.df["ri"]]
        return self

    def removehet(self):
        self.subset(het=False, inplace=True)
        return self

    # @timed
    def subset(
        self,
        chain=None,
        het=None,
        removeres=None,
        atomnames=[],
        chains=[],
        model=None,
        modelidx=None,
        inplace=False,
        removeatoms=[],
        **kw,
    ):
        import numpy as np
        import pandas as pd

        df = self.df
        if chain is not None:
            if isinstance(chain, int):
                chain = list(self.chainseq.keys())[chain]
            df = df.loc[df.ch == ipd.dev.tobytes(chain)]
            # have no idea why, but dataframe gets corrupted  without this
            df = pd.DataFrame(df.to_dict())
            assert len(df) > 0
        if het is False:
            df = df.loc[~df.het]
            df = pd.DataFrame(df.to_dict())
        if het is True:
            df = df.loc[~df.het]
            df = pd.DataFrame(df.to_dict())
        if removeres is not None:
            if isinstance(removeres, (str, bytes)):
                removeres = [removeres]
            for res in removeres:
                res = ipd.dev.tobytes(res)
                df = df.loc[df.rn != res]
                df = pd.DataFrame(df.to_dict())
        if atomnames:
            atomnames = [a.encode() for a in atomnames]
            df = df.loc[np.isin(df.an, atomnames)]
            df = pd.DataFrame(df.to_dict())
        if chains:
            if isinstance(chains, str) and len(chains) == 1:
                chains = [chains]
            chains = [c.encode() for c in chains]
            df = df.loc[np.isin(df.ch, chains)]
            df = pd.DataFrame(df.to_dict())
        if model is not None:
            df = df.loc[df.mdl == model]
            df = pd.DataFrame(df.to_dict())
        if modelidx is not None:
            df = df.loc[df.mdl == self.models()[modelidx]]
            df = pd.DataFrame(df.to_dict())
        if removeatoms:
            idx = np.isin(df.ai, removeatoms)
            # ic(df.loc[idx].an)
            # ic(df.loc[idx].ri)
            df = df.loc[~idx]
            df = pd.DataFrame(df.to_dict())

        if len(df) == 0:
            return None

        df.reset_index(inplace=True, drop=True)
        # assert len(df) > 0

        if inplace:
            self.init(
                df,
                self.meta,
                original_contents=self.original_contents,
                renumber_by_model=True,
            )
            return self
        else:
            return PDBFile(df, meta=self.meta, original_contents=self.original_contents)

    def isonlyaa(self):
        return np.sum(self.het) == 0

    def isonlyhet(self):
        return np.sum(self.het) == len(self.df)

    def models(self):
        return list(np.sort(np.unique(self.df.mdl)))

    def modelidx(self, m):
        models = self.models()
        return models.index(m)

    def dump_pdb(self, fname, filemode=None, **kw):
        filemode = filemode or "w"
        out = open(fname, filemode) if isinstance(fname, (str, bytes)) else fname
        if self.cryst1:
            out.write(self.cryst1 + os.linesep)
        df = self.df
        for i, row in df.iterrows():
            s = ipd.pdb.pdbdump.pdb_format_atom_df(**row, **kw)
            out.write(s)
        if isinstance(fname, (str, bytes)):
            out.write("TER\n")
            out.close()

    dump = dump_pdb

    # @timed
    def atommask(self, atomname, aaonly=True, splitchains=False, quiet=False, **kw):
        assert not splitchains
        if not isinstance(atomname, (str, bytes)):
            return np.stack([self.atommask(a) for a in atomname]).T
        an = atomname.encode() if isinstance(atomname, str) else atomname
        an = an.upper()
        mask = list()
        # ic(self.df.iloc[0])
        for i, (ri, g) in enumerate(self.df.groupby(["ri", "ch"])):
            if np.sum(g.an == an) > 1:
                if not quiet:
                    ic('warning duplicate atom', g, g.an)  # type: ignore
                # assert np.sum(g.an == an) <= 1
            # assert np.sum(g.an == an) <= np.sum(g.an == b'CA') # e.g. O in HOH
            hasatom = np.sum(g.an == an) > 0
            mask.append(hasatom)

        mask = np.array(mask, dtype=bool)
        if aaonly:
            aaonly = self.aamask
            mask = mask[aaonly]
        return mask

    # @timed
    def atomcoords(
        self,
        atomname=["N", "CA", "C", "O", "CB"],
        aaonly=True,
        splitchains=False,
        nomask=False,
        removeempty=False,
        **kw,
    ):
        if splitchains:
            chains = self.splitchains(aaonly=aaonly)
            if removeempty:
                chains = [c for c in chains if c is not None]
            return tuple(zip(*[c.atomcoords(atomname, aaonly, nomask=nomask, **kw) for c in chains]))
        if atomname is None:
            atomname = self.df.an.unique()
        pdb = self
        if not pdb.isonlyaa():
            pdb = pdb.subset(het=False)
        if not isinstance(atomname, (str, bytes)):
            coords, masks = zip(*[pdb.atomcoords(a, aaonly, nomask=nomask, **kw) for a in atomname])  # type: ignore
            # ic(len(coords))
            # ic([len(_) for _ in coords])
            coords = np.stack(coords).swapaxes(0, 1)
            # ic(coords.shape)
            masks = None if nomask else np.stack(masks).T
            return coords, masks

        an = atomname.encode() if isinstance(atomname, str) else atomname
        an = an.upper().strip()
        df = self.df
        idx = self.df.an == an
        df = df.loc[idx]
        while len(df.ri.unique()) != len(df.ri):
            for i in range(1, len(df)):
                if df.ri.iloc[i] == df.ri.iloc[i - 1]:
                    df = df.loc[np.arange(len(df)) != i]
                    break
        xyz = np.stack([df["x"], df["y"], df["z"]]).T
        if nomask:
            return xyz, None
        mask = self.atommask(an, **kw)
        if np.sum(~mask) > 0:
            coords = 9e9 * np.ones((len(mask), 3))
            coords[mask] = xyz
            xyz = coords

        return xyz, mask

    def camask(self, aaonly=False, **kw):
        return self.atommask("ca", aaonly=aaonly)
        # return np.array([np.any(g.an == b'CA') for i, g in self.df.groupby(self.df.ri)])

    def cbmask(self, aaonly=True, **kw):
        return self.atommask("cb", aaonly=aaonly)
        # mask = list()
        # for i, (ri, g) in enumerate(self.df.groupby(self.df.ri)):
        #    assert np.sum(g.an == b'CB') <= 1
        #    assert np.sum(g.an == b'CB') <= np.sum(g.an == b'CA')
        #    hascb = np.sum(g.an == b'CB') > 0
        #    mask.append(hascb)
        # mask = np.array(mask)
        # if aaonly:
        #    aaonly = self.aamask
        #    # ic(aaonly)
        #    mask = mask[aaonly]
        # return mask

    def bb(self, **kw):
        crd, mask = self.atomcoords("n ca c o cb".split(), **kw)
        if isinstance(crd, (tuple, list)):
            crd = np.stack(crd)
        return crd

    def ca(self, **kw):
        crd, mask = self.atomcoords("ca", **kw)
        if isinstance(crd, (tuple, list)):
            crd = np.stack(crd)
        return crd

    def ncac(self, **kw):
        crd, mask = self.atomcoords("n ca c".split(), **kw)
        if isinstance(crd, (tuple, list)):
            crd = np.stack(crd)
        return crd
        # pdb = self.subset(het=False, atomnames=['N', 'CA', 'C'])
        # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']]).T.reshape(-1, 3, 3)
        # return xyz

    def ncaco(self, **kw):
        crd, _mask = self.atomcoords("n ca c o".split(), **kw)
        if isinstance(crd, (tuple, list)):
            crd = np.stack(crd)
        if np.all(crd[..., 3, :3] == 9e9):
            crd = ipd.pdb.add_bb_o_guess(crd[..., :3, :3])
        return crd
        # pdb = self.subset(het=False, atomnames=['N', 'CA', 'C', 'O'])
        # xyz = np.stack([pdb.df['x'], pdb.df['y'], pdb.df['z']])
        # xyz = xyz.T.reshape(-1, 4, 3)
        # return xyz

    def sequence(self):
        return self.seq

    def chain(self, ires):
        ires = ires - 1
        for i, (ch, seq) in enumerate(self.chainseq.items()):
            if ires >= len(seq):
                ires -= len(seq)
            else:
                return i + 1
        else:
            return None
        return self

    def num_chains(self):
        return len(self.chainseq)

    def splitchains(self, aaonly=True):
        chains = list()
        for ch in self.df.ch.unique():
            chains.append(self.subset(chain=ch, het=False if aaonly else None))
        return chains

    def guess_nfold(self):
        assert self.isonlyaa()
        # chains = self.splitchains()
        seq = self.sequence()
        # ic(seq)
        for nfold in range(10, 0, -1):
            if len(seq) % nfold:
                continue
            nres = len(seq) // nfold
            nfoldmatch = [seq[:nres] == seq[ir:ir + nres] for ir in range(nres, len(seq), nres)]
            if all(nfoldmatch):
                return nfold
        assert 0, "nfold 1 should have matched"

    def assign_chains_sym(self):
        nfold = self.guess_nfold()
        if nfold == 1:
            return 1
        nres = self.nres // nfold  # type: ignore
        assert nres * nfold == self.nres  # type: ignore
        res = self.ri.unique()
        for ich in range(nfold):  # type: ignore
            for ires in range(nres * ich, nres * (ich+1)):
                self.df.loc[self.df.ri == res[ires], "ch"] = "ABCDEFGHIJ"[ich].encode()
        return nfold

    def xformed(self, xform, chainA_contacts_only=False, contact_distance=8, startchain=0, **kw):
        kw = ipd.dev.Bunch(kw)
        if xform.squeeze().shape == (4, 4):
            crd = self.coords
            crd2 = ipd.homog.hxform(xform, crd)
            pdb = self.copy()
            pdb.coords = crd2
            chains = list(sorted(set(self.df.ch)))
            for ich, ch in enumerate(chains):
                newch = ipd.pdb.all_pymol_chains[startchain + ich].encode()
                pdb.df.loc[pdb.df.ch == ch, "ch"] = newch
            return pdb
        else:
            import pandas as pd

            assert xform.ndim == 3
            xforms = xform.copy()
            isident = np.all(np.isclose(xform, np.eye(4)), axis=(1, 2))
            if np.any(isident):
                assert np.sum(isident) == 1
                xforms[isident] = xforms[0]
                xforms[0] = np.eye(4)
            chains = list(sorted(set(self.df.ch)))
            chainpdbs = self.splitchains()
            keeppdbs = list()
            if chainA_contacts_only:
                chaincoords = self.ca(splitchains=True)
                chainbodies = [ipd.RigidBody(crd) for crd in chaincoords]
                body = chainbodies[0]
            for imodel, xform in enumerate(xform):
                for ich, ch in enumerate(chains):
                    if chainA_contacts_only:
                        isect = body.intersects(chainbodies[ich], xforms[imodel], mindis=contact_distance)  # type: ignore
                        if not isect:
                            continue
                    newch = ipd.pdb.all_pymol_chains[startchain + len(keeppdbs)].encode()
                    # ic(imodel, ich, ch, newch)
                    xpdb = chainpdbs[ich].xformed(xform)
                    # xpdb.df.loc[xpdb.df.ch == ch, 'ch'] = newch
                    assert len(set(xpdb.df.ch)) == 1
                    xpdb.df.ch = newch
                    xpdb.camask()
                    keeppdbs.append(xpdb)

            df = pd.concat([pdb.df for pdb in keeppdbs])
            newpdb = PDBFile(df, meta=self.meta, original_contents=self.original_contents)
            return newpdb

    @property
    def coords(self):
        crd = ipd.homog.hpoint(np.stack([self.df.x, self.df.y, self.df.z], axis=-1))
        crd.flags.writeable = False
        return crd

    @coords.setter
    def coords(self, coords):
        self.df.x = coords[:, 0]
        self.df.y = coords[:, 1]
        self.df.z = coords[:, 2]

    def sym_chain_groups(pdb, tolerances=_default_tol, **kw):  # type: ignore
        pdb.assign_chains_sym()
        chains = pdb.splitchains()
        groups = list()
        seenit = set()
        for ichain, ch0 in enumerate(chains):
            seq1 = ch0.sequence()
            for jchain in range(ichain):
                seq2 = chains[jchain].sequence()
                matcher = SequenceMatcher(None, seq1, seq2)
                match = matcher.find_longest_match(0, len(seq1), 0, len(seq2))
                if tolerances.seqmatch < 2 * match.size / len(seq1 + seq2):
                    groups.append((ichain, jchain, match))
                    if len(seenit.intersection([ichain, jchain])):
                        raise ValueError("looks like more than 2fold symmetry, not yet supported")
                    seenit.update([ichain, jchain])
        if not groups:
            raise ValueError(f"No symmetrical chains found by longest common substring {tolerances.seqmatch}")
        return chains, groups

    def syminfo(self, tolerances=_default_tol, **kw):
        chains, cgroups = self.sym_chain_groups(tolerances, **kw)
        chaingroup = sorted(cgroups, key=lambda x: x[2].size)[-1]
        ichain, jchain, seqmatch = chaingroup
        ca0 = chains[ichain].ca()[seqmatch.a:seqmatch.a + seqmatch.size]
        ca1 = chains[jchain].ca()[seqmatch.b:seqmatch.b + seqmatch.size]
        rms, _, xrmsfit = ipd.hrmsfit(ca0, ca1)
        if rms > tolerances.rms:
            raise ValueError(f"rmsd {rms:5.3f} between detected symmetric chains is above rms tolerance {tolerances.rms}")
        axis, ang, cen, hel = ipd.haxis_angle_cen_hel_of(xrmsfit)
        if hel > tolerances.translation:
            raise ValueError(f'translation along symaxis of {hel:5.3f} between "symmetric"'
                             " chains is above translation tolerance {tolerances.translation}")
        nfold, ang = _get_nfold_angle(ang, tolerances, **kw)
        assert nfold == 2, f"nfold {nfold} not supported yet"
        return ipd.dev.Bunch(
            axis=axis,
            angle=ang,
            center=cen,
            hel=hel,
            nfold=nfold,
            chaingroups=cgroups,
            chains=chains,
        )

def _atomrecords_to_chainseq(df, ignoremissing=True):
    seq = collections.defaultdict(list)

    prevri = 123456789
    prevhet = False
    for i in range(len(df)):
        ri = df.ri[i]
        if ri == prevri:
            continue
        rn = df.rn[i]
        ch = df.ch[i]
        rn = rn.decode() if isinstance(rn, bytes) else rn
        ch = ch.decode() if isinstance(ch, bytes) else ch
        het = df.het[i]
        if not ignoremissing and not het and not prevhet:
            # mark missing sequence residues if not HET
            for _ in range(prevri + 1, ri):
                seq[ch].append("-")
        prevri = ri
        prevhet = het
        if het:
            if not rn == "HOH":
                seq[ch].append("Z")
            continue
        try:
            seq[ch].append(ipd.pdb.aa321[rn])
        except KeyError:
            seq[ch].append("X")
    return {c: str.join("", s) for c, s in seq.items()}

def join(pdbfiles, chains=None):
    import pandas as pd

    chains = chains or ipd.pdb.all_pymol_chains
    dfs = [x.df for x in pdbfiles]
    ichain = 0
    for i, df in enumerate(dfs):
        for j, c in enumerate(df.ch.unique()):
            df.loc[df.ch == c, "ch"] = chains[ichain].encode()
            ichain += 1
    df = pd.concat(dfs)
    return PDBFile(df, meta=pdbfiles[0].meta, original_contents=pdbfiles[0].original_contents)

def _get_nfold_angle(ang, tolerances, candidates=[2, 3, 4, 5, 6], **kw):
    for nfold in candidates:
        angnf = 2 * np.pi / nfold
        if angnf - tolerances.angle < ang < angnf + tolerances.angle:
            return nfold, ang
    raise ValueError(
        f"Angle {np.degrees(ang)} deviates from any nfold in {candidates} by more than {np.degrees(tolerances.angle)} degrees"
    )

import attrs
import numpy as np
import ipd

bs = ipd.lazyimport('biotite.structure')

def stub(atoms):
    cen = bs.mass_center(atoms)
    _, sigma, components = np.linalg.svd(atoms.coord[atoms.atom_name == 'CA'] - cen)
    return ipd.homog.hframe(*components.T, cen)

def find_frames_by_seqaln_rmsfit(atomslist, tol=0.7, result=None, idx=None, **kw):
    if isinstance(atomslist, bs.AtomArray): atomslist = ipd.atom.split_chains(atomslist)
    tol = kw['tol'] = ipd.Tolerances(tol)
    if result is None: result = FrameSearchResult([], [], [], [], atomslist, tol)
    ca = [a[a.atom_name == 'CA'] for a in atomslist]
    ca = [ipd.atom.split_chains(casub, minlen=20) for casub in ca]
    ca = ipd.dev.addreduce(ca)
    if idx is None: idx = np.arange(len(ca))
    frames, rmsds, matches = [np.eye(4)], [0], [1]
    for i, ca_i_ in enumerate(ca[1:]):
        _, match, matchfrac = ipd.atom.seqalign(ca[0], ca_i_)
        xyz1 = ca[0].coord[match[:, 0]]
        xyz2 = ca_i_.coord[match[:, 1]]
        rms, _, xfit = ipd.homog.hrmsfit(xyz1, xyz2)
        frames.append(xfit), rmsds.append(rms), matches.append(matchfrac)
    assert len(frames) == len(ca)
    frames, rmsds, matches = np.stack(frames), np.array(rmsds), np.array(matches)
    ok = (rmsds < tol.rms_fit) & (matches > tol.seq_match)
    result.add(list(idx), frames[ok], matches[ok], rmsds[ok])
    if all(ok): return result
    unfound = [a for i, a in enumerate(ca) if not ok[i]]
    return find_frames_by_seqaln_rmsfit(unfound, result=result, idx=idx[ok], **kw)

@ipd.dev.subscripable_for_attributes
@attrs.define
class FrameSearchResult():
    frames: list[np.ndarray]
    seqmatch: list[float]
    rmsd: list[float]
    idx: list[list[int]]
    source: object
    Tolerances: ipd.Tolerances

    def __post_init__(self):
        self.ninput = len(self.source)
        self.nentity = len(self.frames)
        self.nentity = sum(len(f > 1) for f in self.frames)

    def add(self, idx, frames, seqmatch, rmsd):
        self['idx'].append(idx)
        self['frames'].append(frames)
        self['seqmatch'].append(seqmatch)
        self['rmsd'].append(rmsd)

    def __repr__(self):
        with ipd.dev.capture_stdio() as printed:
            with ipd.dev.np_compact(4):
                print('FrameSearchResult:')
                print('  frames:', [f.shape for f in self.frames])
                print('  seqmatch:', self.seqmatch)
                print('  rmsd:', self.rmsd)
                print('  idx:', self.idx)
                print('  source:', type(self.source))
                print(self.Tolerances)
        return printed.read()

    def __iter__(self):
        for frame, match, rms in zip(self.frames, self.seqmatch, self.rmsd):
            yield frame, match, rms

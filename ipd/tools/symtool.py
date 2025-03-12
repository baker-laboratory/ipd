import hashlib
from pathlib import Path
import traceback

import ipd

class SymTool(ipd.tools.IPDTool):
    pass

class TestTool(SymTool):

    def detect(self, fnames: list[str], partition: int = 0, npartitions: int = 1):
        custom_tol = dict(default=1e-1,
                          angle=0.04,
                          helical_shift=4,
                          isect=6,
                          dot_norm=0.07,
                          misc_lineuniq=0.2,
                          rms_fit=3,
                          nfold=0.2)
        tol = ipd.dev.Tolerances(**(ipd.sym.symdetect_default_tolerances | custom_tol))
        for fileordir in map(Path, fnames):
            inputs = [fileordir]
            if fileordir.is_dir():
                inputs = [path for path in fileordir.glob('*.bcif.gz')]
            for i, path in enumerate(inputs):
                pdbcode = path.stem.split('.')[0]
                ppd_partition = hashlib.sha1(pdbcode.encode()).hexdigest()
                ppd_partition = int(f'0x{ppd_partition}', base=16) % npartitions
                if ppd_partition != partition: continue
                _sym_check_file(pdbcode, path, tol)

def _sym_check_file(pdbcode, path, tol):
    symanno = ipd.pdb.sym_annotation(pdbcode)
    for id, symid in zip(symanno.id, symanno.sym):
        if symid == 'C1': continue
        print(f'symdetect {pdbcode} {id:2} {symid:4}', flush=True, end=' ')
        try:
            atoms = ipd.atom.testdata(pdbcode, assembly=id, het=False, path=path.parent)
            sinfo = ipd.sym.detect(atoms, tol=tol)
            if isinstance(sinfo, list):
                print([si.symid for si in sinfo], end=' ')
                sinfo = sinfo[0]
            infer_t = sinfo.pseudo_order // sinfo.order
            if symid != sinfo.symid:
                if not any([
                        symid == 'C2' and sinfo.is_dihedral,
                        symid == 'C2' and sinfo.symid in 'TIO',
                        symid == 'C3' and sinfo.symid in 'TIO',
                        symid == 'C4' and sinfo.symid in 'O',
                        symid == 'C5' and sinfo.symid in 'I',
                ]):
                    print(f'mispredict {sinfo.symid}', end=' ', flush=True)
                else:
                    print('OK', end=' ')
            elif sinfo.t_number != infer_t:
                print(f'bad T {infer_t} {syminfo.t_number}', end=' ', flush=True)
            else:
                print('OK', end='')
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]  # Get the last traceback frame
            print(f"{type(e).__name__}: {e} {tb.filename.split('/')[-1]}:{tb.lineno})", end='')
        print(flush=True)

class BuildTool(SymTool):

    def from_components(self, comps: list[str]):
        print(comps)
        assert 0, 'not implemented'

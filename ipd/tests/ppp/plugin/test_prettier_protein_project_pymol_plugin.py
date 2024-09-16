import pytest
import os
import time
from ipd.ppp.plugin.ppppp.prettier_protein_project_pymol_plugin import *

def main():
    # hack_test_filefetcher()
    run_pymol()
    print('test_ppppp DONE', flush=True)

bigfiles = [
    '/home/sheffler/1pgx.cif',
    '/data/sheffler/project/cfnpack_old/1ctd_allaa.wcsp',
    '/data/sheffler/project/cfnpack_old/1ctd_allaa_ex1.wcsp',
    '/data/sheffler/project/cfnpack_old/1ctd_allaa_ex12.wcsp',
    '/data/sheffler/project/cfnpack_old/1d3z_allaa_ex12.wcsp',
    '/data/sheffler/project/cfnpack_old/1pgx_allaa.wcsp',
    # '/data/sheffler/project/cfnpack_old/1pgx_allaa_ex1.wcsp',
    # '/data/sheffler/project/cfnpack_old/1pgx_allaa_ex12.wcsp',
    '/data/digs/net/scratch/ahern/se3_diffusion/training/fm_tip_resume_24/fm_tip_resume_242023-12-03_14:20:34.490797/rank_0/models/RFD_44.pt',
    '/data/digs/net/scratch/ahern/se3_diffusion/training/fm_tip_resume_24/fm_tip_resume_242023-12-03_14:20:34.490797/rank_0/models/RFD_45.pt',
    '/data/digs/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol0.jsonl',
    '/data/digs/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol1.jsonl',
    # '/data/digs/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol2.jsonl',
    # '/data/digs/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol3.jsonl',
    # '/data/digs/projects/ml/struc2seq/data_for_complexes/datasets/PDB-2021AUG02_res_25H_homo_90_xaa_train_vol4.jsonl',
]

def hack_test_filefetcher():
    print('start', flush=True)
    ff = BackgroundFileFetcher(bigfiles, shuffle=False)
    # ff = FileFetcher(bigfiles, shuffle=False)
    for f in ff:
        time.sleep(1)
        print(f, flush=True)

def run_pymol():
    # os.environ['QT_QPA_PLATFORM'] = 'xcb'
    pymol = pytest.importorskip('pymol')
    pymol.pymol_argv = ['pymol', '-q']
    pymol.finish_launching()
    # from ipd.ppp.plugin.ppppp import run_ppppp_gui
    # ui = run_ppppp_gui()
    # while time.sleep(1): pass
    # assert 0
    # from ipd.pymol import ppppp

if __name__ == '__main__':
    main()

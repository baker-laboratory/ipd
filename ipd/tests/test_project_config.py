import os

import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_install_ipd_pre_commit_hook(tmpdir):
    os.makedirs(f'{tmpdir}/projdir', exist_ok=True)
    with ipd.dev.cd(f'{tmpdir}/projdir'):
        os.system('git init')
        ipd.dev.install_ipd_pre_commit_hook('.', config=dict(ipd_pre_commit=True))
        assert os.path.exists('.git/hooks/pre-commit')

if __name__ == '__main__':
    main()

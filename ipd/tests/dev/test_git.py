import os
import tempfile
import ipd

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_install_ipd_pre_commit_hook(tmpdir)

def test_install_ipd_pre_commit_hook(tmpdir):
    os.makedirs(f'{tmpdir}/projdir', exist_ok=True)
    with ipd.dev.cd(f'{tmpdir}/projdir'):
        os.system('git init')
        ipd.dev.git.install_ipd_pre_commit_hook('.')
        assert os.path.exists('.git/hooks/pre-commit')

if __name__ == '__main__':
    main()

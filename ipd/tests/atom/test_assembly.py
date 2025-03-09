import pytest

pytest.importorskip('biotite')
import ipd

config_test = ipd.Bunch(
    re_only=[],
    re_exclude=[],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
        # dryrun=True,
    )

def test_assembly_simple():
    asm = ipd.atom.assembly('1qys')

if __name__ == '__main__':
    main()

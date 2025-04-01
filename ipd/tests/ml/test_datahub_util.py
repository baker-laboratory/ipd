import pytest
dh = pytest.importorskip('datahub', reason='datahub tests require the datahub package')
import ipd

config_test = ipd.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )

def test_load_helen_example():
    ipd.ml.extract_helen_example(
        ipd.dev.package_testdata_path('6u9d_Q_1.pickle.gz')
    )

if __name__ == '__main__':
    main()

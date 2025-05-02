import pytest
import evn


def main():
    evn.testing.quicktest(namespace=globals())


bunch = evn.Bunch(
    dot_norm=evn.Bunch(frac=0.174, tol=0.04, total=282, passes=49),
    isect=evn.Bunch(frac=0.149, tol=1.0, total=302, passes=45),
    angle=evn.Bunch(frac=0.571, tol=0.09, total=42, passes=24),
    helical_shift=evn.Bunch(frac=1.0, tol=1.0, total=47, passes=47),
    axistol=evn.Bunch(frac=0.412, tol=0.1, total=17, passes=7),
    nfold=evn.Bunch(frac=1.0, tol=0.2, total=5, passes=5),
    cageang=evn.Bunch(frac=0.5, tol=0.1, total=2, passes=1),
)


def test_make_table_dict_of_dict():
    with evn.capture_stdio() as out:
        evn.print.print_table(bunch)
    printed = out.read()
    assert (printed.strip() == """
┏━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ key           ┃ frac    ┃ tol     ┃ total ┃ passes ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ dot_norm      │   0.174 │   0.040 │  282  │   49   │
│ isect         │   0.149 │   1.000 │  302  │   45   │
│ angle         │   0.571 │   0.090 │   42  │   24   │
│ helical_shift │   1.000 │   1.000 │   47  │   47   │
│ axistol       │   0.412 │   0.100 │   17  │    7   │
│ nfold         │   1.000 │   0.200 │    5  │    5   │
│ cageang       │   0.500 │   0.100 │    2  │    1   │
└───────────────┴─────────┴─────────┴───────┴────────┘
""".strip())


def test_summary_numpy():
    np = pytest.importorskip('numpy')
    assert evn.summary(np.arange(3)) == '[0 1 2]'
    assert evn.summary(np.arange(300)) == 'ndarray[300]'


if __name__ == '__main__':
    main()

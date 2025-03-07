import numpy as np
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
    )

def test_rel_xform_info():
    frame1, frame2 = np.eye(4), np.eye(4)
    ipd.homog.rel_xform_info(frame1, frame2)

if __name__ == '__main__':
    main()

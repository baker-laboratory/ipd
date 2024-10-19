import numpy as np
import willutil as wu
import willutil.homog as hm
import willutil.viz


def main():
    test_line_strips(headless=False)
    test_pymol_viz_example(headless=False)


def test_line_strips(headless=True):
    # pytest.importorskip('pymol')
    coord = np.ones((100, 4))
    for i in range(10):
        coord[i * 10 : (i + 1) * 10, 0] = 10 * i + np.cumsum(np.random.rand(10))
        coord[i * 10 : (i + 1) * 10, 1] = np.cumsum(np.random.rand(10))
        coord[i * 10 : (i + 1) * 10, 2] = np.cumsum(np.random.rand(10))
    wu.viz.showme(coord, islinestrip=True, headless=headless, breaks=10)


def test_pymol_viz_example(headless=True):
    # pytest.importorskip('pymol')
    frame1 = hm.hframe([1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 0, 0])
    rel = hm.hrot([1, 0, 0], 90, [0, 0, 10])
    rel[0, 3] = 3
    frame2 = rel @ frame1
    xinfo = wu.sym.rel_xform_info(frame1, frame2)
    wu.viz.showme(xinfo, headless=headless)


if __name__ == "__main__":
    main()

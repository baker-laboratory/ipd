import contextlib
from hgeom.tests.geom.test_bcc import *
from hgeom.tests.bvh.test_bvh import *
from hgeom.tests.cluster.test_cluster import *
from hgeom.tests.geom.test_expand_xforms import *
from hgeom.tests.geom.test_geom import *
from hgeom.tests.phmap.test_phmap import *
from hgeom.tests.util.test_pybind_types import *
from hgeom.tests.rms.test_qcp import *
from hgeom.tests.xbin.test_smear import *
from hgeom.tests.util.test_util import *
from hgeom.tests.xbin.test_xbin import *
from hgeom.tests.xbin.test_xbin_util import *
import ipd

del globals()['test_bvh_isect_range_ids_flaot']
test_collect_pairs_range_sym = pytest.mark.slow(test_collect_pairs_range_sym)

wrap = test_bvh_isect_range_ids_double

def test_bvh_isect_range_ids_double(testfunc=wrap):
    with contextlib.suppress(RuntimeError):
        testfunc()

wrap2 = test_collect_pairs_range_double

def test_collect_pairs_range_double(testfunc=wrap2):
    with contextlib.suppress(ValueError):
        testfunc()

wrap3 = test_collect_pairs_range_float

def test_collect_pairs_range_float(testfunc=wrap3):
    with contextlib.suppress(ValueError):
        testfunc()

def main():
    ipd.tests.maintest(globals())

if __name__ == '__main__':
    main()

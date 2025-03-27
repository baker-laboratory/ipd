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

def main():
    ipd.tests.maintest(globals())

if __name__ == '__main__':
    main()

import numpy as np

import ipd

def main():
    test_unhashable_set()

def test_unhashable_set():
    for i in range(10):
        a = set(np.random.randint(7, size=2))
        b = set(np.random.randint(7, size=3))
        ua = ipd.dev.UnhashableSet(a)
        ub = ipd.dev.UnhashableSet(b)
        # ipd.icv(a)
        # ipd.icv(b)
        # ipd.icv(a - b)
        # ipd.icv(setminus(a, b))
        assert (a - b) == set(ua.difference(ub))
        # ipd.icv(b.intersection(a))
        # ipd.icv(setisect(b, a))
        assert b.intersection(a) == set(ub.intersection(ua))
        # ipd.icv(a == b)
        # ipd.icv(setequal(a, b))
        assert (a == b) == (ua == ub)

if __name__ == "__main__":
    main()

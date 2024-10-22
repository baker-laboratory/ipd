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
        # ic(a)
        # ic(b)
        # ic(a - b)
        # ic(setminus(a, b))
        assert (a - b) == set(ua.difference(ub))
        # ic(b.intersection(a))
        # ic(setisect(b, a))
        assert b.intersection(a) == set(ub.intersection(ua))
        # ic(a == b)
        # ic(setequal(a, b))
        assert (a == b) == (ua == ub)

if __name__ == "__main__":
    main()

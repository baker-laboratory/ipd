import sys

if 'doctest' not in sys.modules:
    import ipd

    missing = ipd.lazyimport('does_not_exist')
    missing.BOOM

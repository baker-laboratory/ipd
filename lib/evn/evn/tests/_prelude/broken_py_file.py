import sys

if 'doctest' not in sys.modules:
    import evn

    missing = evn.lazyimport('does_not_exist')
    missing.BOOM

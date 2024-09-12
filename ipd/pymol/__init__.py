import contextlib

with contextlib.suppress(ImportError):
    import pymol, sys, os
    ipd_path = os.path.realpath(os.path.dirname(__file__) + '/..')
    if ipd_path not in sys.path: sys.path.append(ipd_path)

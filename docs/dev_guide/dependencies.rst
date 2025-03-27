Dependencies
==============

IPD has kindof a lot of dependencies. Adding more is ok, but prefer to make them optional.

Optional Dependencies
-------------------------
Add to one of the optional groups in `pyproject.toml`, or add a new group. If you add a new optional dependency, consider adding it to the ``dev`` and/or ``all`` groups. If it is a core dependency, or a particularly "heavy" one, add a note below.

Optional dependencies shold always be imported at the top level using myopt = ipd.lazyimport('myopt'). This will prevent the dependency from being loaded unless it is actually used. Tests using the optional dependency should start with ``pytest.importorskip('myopt')``. You can also put that statement at the top of a test module to skip all tests in that module if the dependency is not available.


Core Dependencies
------------------

hgeom
~~~~~~~
Provides bounding volume hierarchies equivariant to homogeneous transforms. These make all kinds of intersection testing on clouds of points super fast, generally O(log N) for structures that don't clash. Also provides a fast vectorized 'dict' called phmap, very fancy binning of homogeneous trasforms, n-dimensional BCC lattices, and a few more goodies. All is written in C++ and wrapped with pybind11.

evn
~~~~~
Next gen python code formatter that is currently vaporware.

numpy
~~~~~~~~
You know what this is.



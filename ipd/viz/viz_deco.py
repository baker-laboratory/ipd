import functools

import ipd

try:
    import pymol  # type: ignore
except ImportError:
    pass

def pymol_scene(func):
    from ipd.viz.pymol_viz import _showme_state

    @functools.wraps(func)
    def wrapper(
        *args,
        name=func.__name__,
        addtocgo=None,
        suspend_updates=True,
        delprev=True,
        state=_showme_state,
        **kw,
    ):
        if suspend_updates: pymol.cmd.set('suspend_updates', 'on')  # type: ignore
        v = pymol.cmd.get_view()  # type: ignore
        if delprev: pymol.cmd.delete(f'{name}*')  # type: ignore
        state["seenit"][name] += 1
        name += "_%i" % state["seenit"][name]

        bunch = func(*args, name=name, state=state, **kw)
        bunch = bunch or ipd.dev.Bunch(cgo=None, _strict=False)  # type: ignore
        if bunch.get('cgo'):
            if addtocgo is not None:
                addtocgo.extend(bunch.cgo)
            else:
                pymol.cmd.load_cgo(bunch.cgo, name)  # type: ignore
        pymol.cmd.set_view(v)  # type: ignore
        if suspend_updates: pymol.cmd.set('suspend_updates', 'off')  # type: ignore
        return bunch.sub(cgo=addtocgo)

    return wrapper

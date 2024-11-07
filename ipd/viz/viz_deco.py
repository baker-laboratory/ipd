import functools

import ipd

try:
    import pymol
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
        if suspend_updates: pymol.cmd.set('suspend_updates', 'on')

        v = pymol.cmd.get_view()
        if delprev: pymol.cmd.delete(f'{name}*')

        state["seenit"][name] += 1
        name += "_%i" % state["seenit"][name]

        bunch = func(*args, name=name, state=state, **kw)
        bunch = bunch or ipd.dev.Bunch(cgo=None, _strict=False)

        if bunch.get('cgo'):
            if addtocgo is not None:
                addtocgo.extend(bunch.cgo)
            else:
                pymol.cmd.load_cgo(bunch.cgo, name)

        pymol.cmd.set_view(v)
        if suspend_updates: pymol.cmd.set('suspend_updates', 'off')
        return bunch.sub(cgo=addtocgo)

    return wrapper

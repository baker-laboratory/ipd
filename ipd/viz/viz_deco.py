import willutil as wu
import functools

try:
    import pymol
except ImportError:
    pass

def pymol_frame(_FUNCTION_):
    from willutil.viz.pymol_viz import _showme_state

    @functools.wraps(_FUNCTION_)
    def wrapper(
        *args,
        name=_FUNCTION_.__name__,
        addtocgo=None,
        suspend_updates=True,
        delprev=False,
        state=_showme_state,
        **kw,
    ):
        if suspend_updates: pymol.cmd.set('suspend_updates', 'on')

        v = pymol.cmd.get_view()
        if delprev: pymol.cmd.delete(f'{name}*')

        state["seenit"][name] += 1
        name += "_%i" % state["seenit"][name]

        bunch = _FUNCTION_(*args, name=name, delprev=delprev, state=state, **kw)
        bunch = bunch or wu.Bunch(cgo=None)

        if bunch.cgo:
            if addtocgo is not None:
                addtocgo.extend(bunch.cgo)
            else:
                pymol.cmd.load_cgo(bunch.cgo, name)

        pymol.cmd.set_view(v)
        if suspend_updates: pymol.cmd.set('suspend_updates', 'off')
        return bunch.sub(cgo=addtocgo)

    return wrapper

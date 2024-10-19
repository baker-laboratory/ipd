import logging

try:
    from ipd.dev.lazy_import import lazyimport
    rp = lazyimport('RestrictedPython', pip=True)

    def safe_eval(code, **kw):
        return eval(rp.compile_restricted(code, filename='<inline code>', mode='eval'), **kw)

    def safe_exec(code, **kw):
        return exec(rp.compile_restricted(code, filename='<inline code>', mode='exec'), **kw)
except ImportError:
    logging.warning(
        'RestrictedPython not installed, exec and eval will be unsafe. pip install RestrictedPython')
    safe_eval = eval
    safe_exec = exec

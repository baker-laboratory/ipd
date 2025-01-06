import logging

try:
    from ipd.lazy_import import lazyimport
    rp = lazyimport('RestrictedPython', pip=True)

    def safe_eval(code, **kw):
        bytecode = rp.compile_restricted(code, filename='<inline code>', mode='eval')
        return eval(bytecode, **kw)

    def safe_exec(code, **kw):
        print(code)
        bytecode = rp.compile_restricted(code, filename='<inline code>', mode='exec')
        return exec(bytecode, **kw)

except ImportError:
    logging.warning('RestrictedPython not installed, exec and eval will be unsafe. pip install RestrictedPython')
    safe_eval = eval  # type: ignore
    safe_exec = exec  # type: ignore

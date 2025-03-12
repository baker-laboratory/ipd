import logging

try:
    from ipd.lazy_import import lazyimport
    rp = lazyimport('RestrictedPython', pip=True)

    def safe_eval(code, *args):
        bytecode = rp.compile_restricted(code, filename='<inline code>', mode='eval')
        return eval(bytecode, *args)

    def safe_exec(code, *args):
        print(code)
        bytecode = rp.compile_restricted(code, filename='<inline code>', mode='exec')
        return exec(bytecode, *args)

except ImportError:
    logging.warning(
        'RestrictedPython not installed, exec and eval will be unsafe. pip install RestrictedPython')
    safe_eval = eval
    safe_exec = exec

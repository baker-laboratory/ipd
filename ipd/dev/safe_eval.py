import logging

try:
    from RestrictedPython import compile_restricted

    def safe_eval(code, **kw):
        return eval(compile_restricted(code, filename='<inline code>', mode='eval'), **kw)

    def safe_exec(code, **kw):
        return exec(compile_restricted(code, filename='<inline code>', mode='exec'), **kw)
except:
    logging.warning('RestrictedPython not installed, exec and eval will be unsafe. pip install RestrictedPython')
    safe_eval = eval
    safe_exec = exec

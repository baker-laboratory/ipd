def map(f, x):
    if isinstance(x, dict):
        return {k: f(v) for k, v in x.items()}
        return [f(v) for v in x]

def visit(f, x):
    if isinstance(x, dict):
        return {k: recursemap(f, v) for k, v in x.items()}
    elif isinstance(x, list):
        return [recursemap(f, v) for v in x]
    else:
        return f(x)

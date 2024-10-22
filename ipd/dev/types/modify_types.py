def list_classes(data):
    seenit = set()

    def visitor(x):
        seenit.add(x.__class__)

    visit(data, visitor)
    return seenit

def change_class(data, clsmap) -> None:
    def visitor(x):
        if x.__class__ in clsmap:
            x.__class__ = clsmap[x.__class__]

    visit(data, visitor)

def visit(data, func) -> None:
    if isinstance(data, dict):
        visit(list(data.keys()), func)
        visit(list(data.values()), func)
    elif isinstance(data, list):
        for x in data:
            visit(x, func)
    else:
        func(data)

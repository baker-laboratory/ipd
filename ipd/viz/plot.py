def scatter(x, y, show=True, **kw):
    from matplotlib import pyplot

    pyplot.scatter(x, y, **kw)
    if show:
        pyplot.show()

def hist(x, show=True, **kw):
    from matplotlib import pyplot

    pyplot.hist(x, **kw)
    if show:
        pyplot.show()

def show():
    from matplotlib import pyplot

    pyplot.show()

def showimage(data, **kw):
    import matplotlib.pyplot as plt
    plt.imshow(data, **kw)
    plt.colorbar()
    plt.show()

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

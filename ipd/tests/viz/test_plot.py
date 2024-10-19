import willutil as wu
import numpy as np


def main():
    x = np.random.normal(size=10)
    y = np.random.normal(size=10)
    print(x)
    print(y)

    wu.viz.scatter(x, y)
    wu.viz.hist(x)

    wu.viz.scatter(x, y, show=False)
    wu.viz.scatter(x + 1, y, show=False)
    wu.viz.hist(x)
    wu.viz.show()


if __name__ == "__main__":
    main()

import numpy as np

import ipd

def main():
    x = np.random.normal(size=10)
    y = np.random.normal(size=10)
    print(x)
    print(y)

    ipd.viz.scatter(x, y)
    ipd.viz.hist(x)

    ipd.viz.scatter(x, y, show=False)
    ipd.viz.scatter(x + 1, y, show=False)
    ipd.viz.hist(x)
    ipd.viz.show()

if __name__ == "__main__":
    main()

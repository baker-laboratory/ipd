class Cylinder(object):
    """Cylinder shape for pymol."""
    def __init__(self, start, end, radius=1, color=[1, 1, 1], radius2=None):
        super(Cylinder, self).__init__()
        self.start = start
        self.end = end
        self.radius = radius
        self.color = color
        radius2 = radius if radius2 is None else radius2

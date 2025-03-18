import ipd

def symmetric_crop(assemb, maxsize=400):
    ichain = assemb.uniform_random_chain()
    neighborhood = assemb.get_neighborhood(ichain)
    assert isinstance(neighborhood, ipd.atom.Assembly)

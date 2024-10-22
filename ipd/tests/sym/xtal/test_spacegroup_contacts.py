import ipd
from ipd.sym.xtal.spacegroup_contacts import minimal_spacegroup_cover_symelems

def main():
    # symelems = ipd.sym.xtal.symelems('P6')
    # genelems = [symelems[i] for i in [0, 1, 3]]
    # complete = check_if_symelems_complete('P6', genelems)

    # minimal_spacegroup_cover_symelems('P6')

    covers = ipd.load("spacegroup_covers_2elem.pickle")
    for sg in ipd.sym.xtal.sg_all_chiral:
        if sg in covers:
            continue
        covers[sg] = minimal_spacegroup_cover_symelems(sg, maxelems=2)
        # ic(covers)
        ipd.save(covers, "spacegroup_covers_2elem.pickle")
        # minimal_spacegroup_cover_symelems(sg, maxelems=2, noscrew=True, nocompound=True)

    for sg, elemslist in covers.items():
        elemslist = [
            elems for elems in elemslist if all([
                # all(e.iscyclic for e in elems),
                any(e.iscyclic for e in elems),
                any(e.isscrew for e in elems),
            ])
        ]
        if elemslist:
            ipd.printheader(sg)
        for elems in elemslist:
            for e in elems:
                print(e)
            print()

# def test_sg_color_symel

if __name__ == "__main__":
    main()

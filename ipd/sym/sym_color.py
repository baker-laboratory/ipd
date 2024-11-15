import ipd

th = ipd.lazyimport('torch')

def make_sequential_colors(colors):
    """This function takes a list of colors and returns a new list of colors
    where the colors are sequential.

    For example, if the input list is [7, 7, 2, 2, 3, 3], the output
    list will be [0, 0, 1, 1, 2, 2].
    """
    assert len(colors)
    colors = th.as_tensor(colors, dtype=int)
    colors += 1000000
    newcolors = colors.clone()
    col = 0
    for i in range(len(colors) - 1):
        newcolors[i] = col
        if colors[i] != colors[i + 1]:
            col += 1
    newcolors[-1] = col
    return newcolors

def symslices_from_colors(
    nsub: int,
    colors: 'th.Tensor',  # type: ignore
    isasu: bool = True,
    Lasu: 'th.Tensor' = None,  # type: ignore
    recolor: bool = True,
):
    """This function takes a list of colors and returns a list of slices.

    The slices are such that
    the colors within each slice are the same. The slices are returned as a list of tuples. Each tuple
    contains the total number of colors in the list, the start index of the slice, and the end
    index of the slice. For example, if the input list is [7, 7, 2, 2, 3, 3], the output list will
    be [(6, 0, 2), (6, 2, 4), (6, 4, 6)]. If isasu is True, the slice sizes are incresed by a factor
    of nsub. If Lasu is not None, the slice sizes are set to Lasu*nsub, leaving room for unsymmetric
    entries. If recolor is True, the colors are made sequential before the slices are calculated.

    Args:
        - nsub: int, number of subunits
        - colors: Tensor of int, 'colors' representing different regions of a structure
        - isasu: bool, if True, the slice sizes are incresed by a factor of nsub
        - Lasu: Tensor of int, the size of the slice asu for each color. if lasu*nsub dosent fill
            the whole region, there will be space before the next slice
        - recolor: bool, if True, the colors are made sequential before the slices are calculated
    Returns:
        slices: list of tuples, each tuple contains the total number of colors in the list, the start
            index of the slice, and the end index of the slice
    """
    ignore = colors < 0
    # if th.any(ignore) and isasu:
    # raise NotImplementedError('ignoring negative colors not implemented for isasu=True')
    if recolor: colors = make_sequential_colors(colors)
    if Lasu is None: Lasu = th.full((colors[-1] + 1, ), -1)
    if isasu:
        Lasym = th.tensor([th.sum(colors == i) for i in range(colors[-1] + 1)], dtype=int)
        Lasu = th.where(Lasu < 0, Lasym, Lasu)
        Lunsym = Lasym - Lasu
        Lsym = Lasu * nsub
        L = Lsym + Lunsym
        assert th.sum(Lasym) == len(colors)
    else:
        L = th.tensor([th.sum(colors == i) for i in range(colors[-1] + 1)], dtype=int)
        Lasu = th.where(Lasu < 0, L // nsub, Lasu)
        Lsym = Lasu * nsub
        Lunsym = L - Lsym
        Lasym = Lasu + Lunsym
        assert th.sum(L) == len(colors)

    if 0:
        print(isasu)
        print(colors)
        print('L      ', L)
        print('Lsym   ', Lsym)
        print('Lasu   ', Lasu)
        print('Lunsym ', Lunsym)
        print('Lasym  ', Lasym)

    assert th.all(Lasym >= 0)
    assert th.all(Lsym[Lsym != 0] % Lasu[Lsym != 0] == 0)
    assert th.all(Lsym == Lasu * nsub)
    assert th.all(L >= Lsym)
    assert th.all(L == Lsym + Lunsym)

    Ltot = int(th.sum(L))
    slices = list()
    start = 0
    for i, (l, lsym) in enumerate(zip(L, Lsym)):
        # ic(len(colors), start)
        slice = (Ltot, start, start + (0 if not isasu and ignore[start] else int(lsym)))
        if isasu or not ignore[slice[1]]: slices.append(slice)
        start += int(l)
    if isasu and slices:
        idx = ipd.sym.SymIndex(nsub, slices)
        ignoresym = th.zeros(idx.L, dtype=bool)
        # ic(idx.L, Ltot, slices)
        ignoresym[idx.idx_asym_to_sym] = ignore
        slices, slices2 = list(), slices
        for s in slices2:
            if not ignoresym[s[1]]: slices.append(s)
    return slices

import ipd

th = ipd.lazyimport('torch')

def make_sequential_colors(colors):
    """This function takes a list of colors and returns a new list of colors
    where the colors are sequential.

    For example, if the input list is [7, 7, 2, 2, 3, 3], the output
    list will be [0, 0, 1, 1, 2, 2].
    """
    assert len(colors)
    colors = th.as_tensor(colors, dtype=int).clone()
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
    colors: 'th.Tensor',
    isasym: bool = True,
    Lasu: 'th.Tensor' = None,
    recolor: bool = True,
    debug: bool = False,
):
    """This function takes a list of colors and returns a list of slices.

    The slices are such that
    the colors within each slice are the same. The slices are returned as a list of tuples. Each tuple
    contains the total number of colors in the list, the start index of the slice, and the end
    index of the slice. For example, if the input list is [7, 7, 2, 2, 3, 3], the output list will
    be [(6, 0, 2), (6, 2, 4), (6, 4, 6)]. If isasym is True, the slice sizes are incresed by a factor
    of nsub. If Lasu is not None, the slice sizes are set to Lasu*nsub, leaving room for unsymmetric
    entries. If recolor is True, the colors are made sequential before the slices are calculated.

    Args:
        - nsub: int, number of subunits
        - colors: Tensor of int, 'colors' representing different regions of a structure
        - isasym: bool, if True, the slice sizes are incresed by a factor of nsub
        - Lasu: Tensor of int, the size of the slice asu for each color. if lasu*nsub dosent fill
            the whole region, there will be space before the next slice
        - recolor: bool, if True, the colors are made sequential before the slices are calculated
    Returns:
        slices: list of tuples, each tuple contains the total number of colors in the list, the start
            index of the slice, and the end index of the slice
    """
    ignore = colors < 0
    ignore_slice = None
    # if th.any(ignore) and isasym:
    # raise NotImplementedError('ignoring negative colors not implemented for isasym=True')
    if recolor:
        colors = make_sequential_colors(colors)
    if Lasu is None:
        Lasu = th.full((colors[-1] + 1, ), -1)
    if isasym:
        Lasym = th.tensor([th.sum(colors == i) for i in range(colors[-1] + 1)], dtype=int)
        assert th.sum(Lasym) == len(colors)
        istart = [0] + th.cumsum(Lasym, 0).tolist()[:-1]
        ignore_slice = th.as_tensor([ignore[i] for i in istart])
        Lasu = th.where(Lasu < 0, Lasym, Lasu)
        Lasu = Lasu * ~ignore_slice
        Lunsym = Lasym - Lasu
        Lsym = Lasu * nsub * ~ignore_slice
        L = Lsym + Lunsym
    else:
        L = th.tensor([th.sum(colors == i) for i in range(colors[-1] + 1)], dtype=int)
        Lasu = th.where(Lasu < 0, L // nsub, Lasu)
        Lsym = Lasu * nsub
        Lunsym = L - Lsym
        Lasym = Lasu + Lunsym
        assert th.sum(L) == len(colors)

    if debug:
        print(isasym, flush=True)
        print(colors, flush=True)
        print(ignore_slice, flush=True)
        print('L      ', L, flush=True)
        print('Lsym   ', Lsym, flush=True)
        print('Lasu   ', Lasu, flush=True)
        print('Lunsym ', Lunsym, flush=True)
        print('Lasym  ', Lasym, flush=True)

    assert th.all(Lasym >= 0)
    assert th.all(Lsym[Lsym != 0] % Lasu[Lsym != 0] == 0)
    assert th.all(Lsym == Lasu * nsub)
    assert th.all(L >= Lsym)
    assert th.all(L == Lsym + Lunsym)

    Ltot = int(th.sum(L))
    slices = list()
    start = 0
    for i, (l, lsym) in enumerate(zip(L, Lsym)):
        # ipd.icv(len(colors), start)
        slice = (Ltot, start, start + (0 if not isasym and ignore[start] else int(lsym)))
        if ignore_slice is not None:
            if not ignore_slice[i]: slices.append(slice)
        elif ignore[slice[1]]: slices.append(slice)
        start += int(l)
    if isasym and slices:
        idx = ipd.sym.SymIndex(nsub, slices)
        ignoresym = th.zeros(idx.L, dtype=bool)
        # ipd.icv(idx.L, Ltot, slices)
        ignoresym[idx.idx_asym_to_sym] = ignore
        slices, slices2 = list(), slices
        for s in slices2:
            if not ignoresym[s[1]]: slices.append(s)

    return slices

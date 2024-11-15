def sort_ncac_coords(coords):
    # sort atom pairs within residues
    for i, j in [
        (0, 1),  # N Ca
        (1, 2),  # Ca C
        (1, 4),  # Ca Cb
        (2, 3),  # C O
    ]:
        dist = np.linalg.norm(coords[None, :, i] - coords[:, None, j], axis=-1)  # type: ignore
        # np.fill_diagonal(dist, 9e9)
        closeres = np.argmin(dist, axis=0)  # type: ignore
        # print(np.min(dist, axis=0))
        coords[:, j] = coords[closeres, j]
        # print(np.linalg.norm(coords[:, i] - coords[:, j], axis=-1))
        assert np.all(np.linalg.norm(coords[:, i] - coords[:, j], axis=-1) < 2)  # type: ignore
    # now permute whole res to N-C is connected
    dist = np.linalg.norm(coords[None, :, 2] - coords[:, None, 0], axis=-1)  # type: ignore
    # np.fill_diagonal(dist, 9e9)
    nextres = np.argmin(dist, axis=0)[:-1]  # type: ignore
    newcoords = [coords[0]]
    prev = 0
    for i in range(len(nextres)):
        prev = nextres[prev]
        newcoords.append(coords[prev])
    newcoords = np.stack(newcoords)  # type: ignore
    coords = newcoords
    print(coords.shape)
    distnc = np.linalg.norm(newcoords[:-1, 2] - newcoords[1:, 0], axis=-1)  # type: ignore
    assert np.all(distnc < 2)  # type: ignore
    # ipd.showme(coords[:, :2], islinestrip=True)
    dist2 = np.linalg.norm(newcoords[None, :, 2] - newcoords[:, None, 0], axis=-1)  # type: ignore
    np.fill_diagonal(dist2, 9e9)  # type: ignore
    nextres2 = np.argmin(dist2, axis=0)[:-1]  # type: ignore
    assert np.allclose(nextres2, np.arange(1, len(nextres2) + 1))  # type: ignore
    ipd.showme(coords[:, :4], islinestrip=True)  # type: ignore
    return newcoords

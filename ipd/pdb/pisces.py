import os

import deferred_import  # type: ignore

pd = deferred_import.deferred_import("pandas")
import ipd

def get_pisces_set(maxresl, max_seq_ident, **kw):
    f = get_pisces_file(max_seq_ident=max_seq_ident, maxresl=maxresl, **kw)
    return read_pisces(f, **kw)

def read_pisces(setname, **kw):
    print("read_pisces", setname)
    fname = ipd.dev.package_data_path(f"pdb/pisces/{setname}.xz")
    with ipd.dev.open_lzma_cached(fname) as inp:
        df = pd.read_fwf(inp)
    df["PDBchain"] = df["PDBchain"].astype("S5")
    df["code"] = [pc[:4] for pc in df["PDBchain"]]
    df["code"] = df["PDBchain"].astype("S4")
    df["chain"] = [pc[4:5] for pc in df["PDBchain"]]
    df["chain"] = df["chain"].astype("S1")
    df["method"] = df["method"].astype("S5")
    df.set_index(df["PDBchain"], inplace=True)
    return df

def get_pisces_file(max_seq_ident=50, maxresl=1.5, **kw):
    "get largest pisces set that meets requirements"
    lastkey = None
    for si, resl in picesfiles:
        if si >= max_seq_ident and resl > maxresl:
            return picesfiles[lastkey]  # type: ignore
        lastkey = si, resl
    raise ValueError(f"cant find pisces set for si={max_seq_ident}, maxresl={maxresl}")

piscesurl = "https://dunbrack.fccc.edu/pisces/download/"
picesfiles = {
    (15, 1.0): "cullpdb_pc15.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains256",
    (15, 1.2): "cullpdb_pc15.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains687",
    (15, 1.5): "cullpdb_pc15.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains1718",
    (15, 1.8): "cullpdb_pc15.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains2846",
    (15, 2.0): "cullpdb_pc15.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains3582",
    (15, 2.2): "cullpdb_pc15.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains4046",
    (15, 2.5): "cullpdb_pc15.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains4574",
    (15, 2.8): "cullpdb_pc15.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains4876",
    (15, 3.0): "cullpdb_pc15.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains5042",
    (20, 1.0): "cullpdb_pc20.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains285",
    (20, 1.2): "cullpdb_pc20.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains868",
    (20, 1.5): "cullpdb_pc20.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains2392",
    (20, 1.8): "cullpdb_pc20.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains4222",
    (20, 2.0): "cullpdb_pc20.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains5383",
    (20, 2.2): "cullpdb_pc20.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains6108",
    (20, 2.5): "cullpdb_pc20.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains6981",
    (20, 2.8): "cullpdb_pc20.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains7469",
    (20, 3.0): "cullpdb_pc20.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains7708",
    (25, 1.0): "cullpdb_pc25.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains311",
    (25, 1.2): "cullpdb_pc25.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1041",
    (25, 1.5): "cullpdb_pc25.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains3253",
    (25, 1.8): "cullpdb_pc25.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains6131",
    (25, 2.0): "cullpdb_pc25.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains8109",
    (25, 2.2): "cullpdb_pc25.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains9314",
    (25, 2.5): "cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains10682",
    (25, 2.8): "cullpdb_pc25.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains11493",
    (25, 3.0): "cullpdb_pc25.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains11822",
    (30, 1.0): "cullpdb_pc30.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains336",
    (30, 1.2): "cullpdb_pc30.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1161",
    (30, 1.5): "cullpdb_pc30.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains3958",
    (30, 1.8): "cullpdb_pc30.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains8074",
    (30, 2.0): "cullpdb_pc30.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains10981",
    (30, 2.2): "cullpdb_pc30.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains12746",
    (30, 2.5): "cullpdb_pc30.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains14717",
    (30, 2.8): "cullpdb_pc30.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains15857",
    (30, 3.0): "cullpdb_pc30.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains16322",
    (40, 1.0): "cullpdb_pc40.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains367",
    (40, 1.2): "cullpdb_pc40.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1358",
    (40, 1.5): "cullpdb_pc40.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains4938",
    (40, 1.8): "cullpdb_pc40.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains10839",
    (40, 2.0): "cullpdb_pc40.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains15334",
    (40, 2.2): "cullpdb_pc40.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains18147",
    (40, 2.5): "cullpdb_pc40.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains21333",
    (40, 2.8): "cullpdb_pc40.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains23120",
    (40, 3.0): "cullpdb_pc40.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains23880",
    (50, 1.0): "cullpdb_pc50.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains389",
    (50, 1.2): "cullpdb_pc50.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1476",
    (50, 1.5): "cullpdb_pc50.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains5505",
    (50, 1.8): "cullpdb_pc50.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains12541",
    (50, 2.0): "cullpdb_pc50.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains18143",
    (50, 2.2): "cullpdb_pc50.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains21718",
    (50, 2.5): "cullpdb_pc50.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains25803",
    (50, 2.8): "cullpdb_pc50.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains28115",
    (50, 3.0): "cullpdb_pc50.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains29087",
    (60, 1.0): "cullpdb_pc60.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains401",
    (60, 1.2): "cullpdb_pc60.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1552",
    (60, 1.5): "cullpdb_pc60.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains5860",
    (60, 1.8): "cullpdb_pc60.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains13669",
    (60, 2.0): "cullpdb_pc60.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains20003",
    (60, 2.2): "cullpdb_pc60.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains24097",
    (60, 2.5): "cullpdb_pc60.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains28855",
    (60, 2.8): "cullpdb_pc60.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains31563",
    (60, 3.0): "cullpdb_pc60.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains32707",
    (70, 1.0): "cullpdb_pc70.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains412",
    (70, 1.2): "cullpdb_pc70.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1611",
    (70, 1.5): "cullpdb_pc70.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains6149",
    (70, 1.8): "cullpdb_pc70.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains14571",
    (70, 2.0): "cullpdb_pc70.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains21452",
    (70, 2.2): "cullpdb_pc70.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains25965",
    (70, 2.5): "cullpdb_pc70.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains31238",
    (70, 2.8): "cullpdb_pc70.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains34278",
    (70, 3.0): "cullpdb_pc70.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains35564",
    (80, 1.0): "cullpdb_pc80.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains418",
    (80, 1.2): "cullpdb_pc80.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1642",
    (80, 1.5): "cullpdb_pc80.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains6397",
    (80, 1.8): "cullpdb_pc80.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains15352",
    (80, 2.0): "cullpdb_pc80.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains22736",
    (80, 2.2): "cullpdb_pc80.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains27634",
    (80, 2.5): "cullpdb_pc80.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains33395",
    (80, 2.8): "cullpdb_pc80.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains36794",
    (80, 3.0): "cullpdb_pc80.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains38233",
    (90, 1.0): "cullpdb_pc90.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains425",
    (90, 1.2): "cullpdb_pc90.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1686",
    (90, 1.5): "cullpdb_pc90.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains6702",
    (90, 1.8): "cullpdb_pc90.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains16353",
    (90, 2.0): "cullpdb_pc90.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains24413",
    (90, 2.2): "cullpdb_pc90.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains29798",
    (90, 2.5): "cullpdb_pc90.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains36260",
    (90, 2.8): "cullpdb_pc90.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains40217",
    (90, 3.0): "cullpdb_pc90.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains41933",
    (95, 1.0): "cullpdb_pc95.0_res0.0-1.0_len40-10000_R0.2_Xray_d2021_11_19_chains432",
    (95, 1.2): "cullpdb_pc95.0_res0.0-1.2_len40-10000_R0.2_Xray_d2021_11_19_chains1725",
    (95, 1.5): "cullpdb_pc95.0_res0.0-1.5_len40-10000_R0.25_Xray_d2021_11_19_chains6901",
    (95, 1.8): "cullpdb_pc95.0_res0.0-1.8_len40-10000_R0.25_Xray_d2021_11_19_chains17030",
    (95, 2.0): "cullpdb_pc95.0_res0.0-2.0_len40-10000_R0.25_Xray_d2021_11_19_chains25625",
    (95, 2.2): "cullpdb_pc95.0_res0.0-2.2_len40-10000_R0.25_Xray_d2021_11_19_chains31432",
    (95, 2.5): "cullpdb_pc95.0_res0.0-2.5_len40-10000_R0.3_Xray_d2021_11_19_chains38520",
    (95, 2.8): "cullpdb_pc95.0_res0.0-2.8_len40-10000_R0.3_Xray_d2021_11_19_chains42980",
    (95, 3.0): "cullpdb_pc95.0_res0.0-3.0_len40-10000_R0.3_Xray_d2021_11_19_chains44920",
}

def download_pisces(path=None):
    if path is None:
        path = ipd.dev.package_data_path("pdb/pisces")
    os.makedirs(path, exist_ok=True)  # type: ignore
    import urllib

    for piscesfile in picesfiles.values():
        url = piscesurl + "/" + piscesfile
        fname = os.path.join(path, piscesfile)  # type: ignore
        if not os.path.exists(fname + ".xz"):
            print("downloading", piscesfile)
            urllib.request.urlretrieve(url, fname)  # type: ignore
            os.system("xz " + fname)
        else:
            print("already downloaded", piscesfile)

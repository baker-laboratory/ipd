import collections
import gzip
import logging
import lzma
import os
import sys
import urllib.request

import ipd

log = logging.getLogger(__name__)

class PDBSearchResult:
    def __init__(self, pdbs):
        self.pdbs = pdbs

class PDBMetadata:
    # __all__ attribute required for class to belave as module
    __all__ = list(set(vars().keys()) - {"__module__", "__qualname__"})

    @property
    def resl(self):
        return self._load_cached("resl", self.load_pdb_resl_data)

    @property
    def xtal(self):
        return self._load_cached("xtal", self.load_pdb_xtal_data)

    @property
    def chainseq(self):
        return self._load_cached("chainseq", self.load_pdb_seqres_data)

    @property
    def seq(self):
        return self._load_cached("seq", self.get_full_seq)

    @property
    def nres(self):
        return self._load_cached("nres", self.get_nres)

    @property
    def compound(self):
        return self._load_cached("compound", self.get_compound)

    @property
    def biotype(self):
        return self._load_cached("biotype", self.get_biotype)

    @property
    def entrytype(self):
        return self._load_cached("entrytype", self.get_entrytype)

    @property
    def byentrytype(self):
        return self._load_cached("byentrytype", self.get_byentrytype)

    @property
    def source(self):
        return self._load_cached("source", self.get_source)

    @property
    def rescount(self):
        return self._load_cached("rescount", self.get_rescount)

    @property
    def ligcount(self):
        return self._load_cached("ligcount", self.get_ligcount)

    @property
    def ligpdbs(self):
        return self._load_cached("ligpdbs", self.get_ligpdbs)

    @property
    def clust30(self):
        return self._load_cached("clust40", lambda: self.get_clust("30"))

    @property
    def clust40(self):
        return self._load_cached("clust40", lambda: self.get_clust("40"))

    @property
    def clust50(self):
        return self._load_cached("clust50", lambda: self.get_clust("50"))

    @property
    def clust70(self):
        return self._load_cached("clust70", lambda: self.get_clust("70"))

    @property
    def clust90(self):
        return self._load_cached("clust90", lambda: self.get_clust("90"))

    @property
    def clust95(self):
        return self._load_cached("clust95", lambda: self.get_clust("95"))

    @property
    def clust100(self):
        return self._load_cached("clust100", lambda: self.get_clust("100"))

    def make_pdb_set(
        self,
        maxresl=2.0,
        minres=50,
        maxres=500,
        max_seq_ident=0.5,
        pisces_chains=True,
        entrytype="prot",
    ):
        # print(minres, maxres, maxres, max_seq_ident)
        piscesdf = ipd.pdb.get_pisces_set(maxresl, max_seq_ident)
        pisces = set(_.decode() for _ in piscesdf.code)
        if max_seq_ident <= 1.0:
            max_seq_ident *= 100

        maxresok = set(self.nres.index[(self.nres <= maxres)])
        minresok = set(self.nres.index[(self.nres >= minres)])
        reslok = set(self.resl.index[self.resl <= maxresl])
        allok = minresok.intersection(maxresok.intersection(reslok))
        if entrytype.upper() not in "ANY ALL".split():
            entrytypeok = self.byentrytype[entrytype]
            allok = allok.intersection(entrytypeok)
        hits = allok.intersection(pisces)

        print("==== make_pdb_set stats ====")
        print("maxresok", len(maxresok))
        print("minresok", len(minresok))
        print("reslok", len(reslok))
        print("allok", len(allok))
        print("pisces", len(pisces))
        print("hits", len(hits))

        if pisces_chains:
            # return all pisces chains rather than pdb codes
            hits = {h.encode() for h in hits}
            chains = piscesdf.chain.unique()
            pdbchains = set(piscesdf.PDBchain)
            chainhits = set()
            for c in chains:
                chits = set([_ + c for _ in hits])
                chits &= pdbchains
                chainhits.update(chits)
            hits = {h.decode() for h in chainhits}
            print("chainhits", len(hits))

        print("============================")
        return hits

    def __init__(self):
        self.urls = ipd.dev.Bunch(
            author="https://ftp.wwpdb.org/pub/pdb/derived_data/index/author.idx",
            compound="https://ftp.wwpdb.org/pub/pdb/derived_data/index/compound.idx",
            resl="https://ftp.wwpdb.org/pub/pdb/derived_data/index/resolu.idx",
            xtal="https://ftp.wwpdb.org/pub/pdb/derived_data/index/crystal.idx",
            entries="https://ftp.wwpdb.org/pub/pdb/derived_data/index/entries.idx",
            onhold="https://ftp.wwpdb.org/pub/pdb/derived_data/index/on_hold.list",
            entrytypes="https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt",
            seqres="https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz",
            source="https://ftp.wwpdb.org/pub/pdb/derived_data/index/source.idx",
            clust30="https://cdn.rcsb.org/resources/sequence/clusters/bc-30.out",
            clust40="https://cdn.rcsb.org/resources/sequence/clusters/bc-40.out",
            clust50="https://cdn.rcsb.org/resources/sequence/clusters/bc-50.out",
            clust70="https://cdn.rcsb.org/resources/sequence/clusters/bc-70.out",
            clust90="https://cdn.rcsb.org/resources/sequence/clusters/bc-90.out",
            clust95="https://cdn.rcsb.org/resources/sequence/clusters/bc-95.out",
            clust100="https://cdn.rcsb.org/resources/sequence/clusters/bc-100.out",
        )
        self.metadata = ipd.dev.Bunch()
        # self.metadata = ipd.dev.Bunch(_strict=True)

    def get_full_seq(self):
        chainseq = self.chainseq
        seq = dict()
        for code, seqs in chainseq.items():
            seq[code] = str.join("", seqs.values())
        return seq

    def _load_cached(self, name, loadfunc):
        if name not in self.metadata:
            try:
                val = ipd.load_package_data(f"pdb/meta/{name}.pickle")
            except FileNotFoundError:
                val = loadfunc()
                ipd.dev.save_package_data(val, f"pdb/meta/{name}.pickle")
            self.metadata[name] = val
        return self.metadata[name]

    def update_source_files(self, replace=True):
        for name, url in self.urls.items():
            fname = ipd.dev.package_data_path(f"pdb/meta/{name}.txt")
            if not replace and os.path.exists(fname + ".xz"):  # type: ignore
                continue
            if name == "seqres":
                fname += ".gz"  # type: ignore
            urllib.request.urlretrieve(url, fname)
            log.info(f"downloading {fname}")
            assert os.path.exists(fname)  # type: ignore
        # recompress seqres
        fn = ipd.dev.package_data_path("pdb/meta/seqres.txt")
        if not os.path.exists(fn + ".xz"):  # type: ignore
            with gzip.open(fn + ".gz") as inp:  # type: ignore
                with lzma.open(fn + ".xz", "wb") as out:  # type: ignore
                    out.write(inp.read())
            os.remove(fn + ".gz")  # type: ignore
        for name in (
                "author",
                "compound",
                "resl",
                "xtal",
                "entries",
                "onhold",
                "entrytypes",
                "source",
                "clust30",
                "clust40",
                "clust50",
                "clust70",
                "clust90",
                "clust95",
                "clust100",
        ):
            fname = ipd.dev.package_data_path(f"pdb/meta/{name}.txt")
            if os.path.exists(fname):  # could skipped download  # type: ignore
                log.info(f"running xz {fname}")
                os.system(f"xz {fname}")

    def clear_pickle_cache(self, names):
        if isinstance(names, str):
            names = [names]
        for name in names:
            fn = ipd.dev.package_data_path(f"pdb/meta/{name}.pickle")
            if os.path.exists(fn):  # type: ignore
                os.remove(fn)  # type: ignore
            if name in self.metadata:
                del self.metadata[name]

    def load_pdb_xtal_data(self):
        xtal = dict()
        count = 0
        with ipd.open_package_data("pdb/meta/xtal.txt.xz") as inp:
            for line in inp:
                count += 1
                if count < 5:
                    continue
                line = line.decode().strip()
                code = line[:4]
                cryst1 = line[5:].strip()
                xtal[code] = cryst1
        return xtal

    def load_pdb_seqres_data(self):
        pdbseq = collections.defaultdict(dict)
        pdb = None
        with ipd.open_package_data("pdb/meta/seqres.txt.xz") as inp:
            for line in inp:
                line = line.decode()
                if line.startswith(">"):
                    pdb = line[1:7]
                    # print(pdb)
                else:
                    code = pdb[:4].upper()  # type: ignore
                    chain = pdb[5]  # type: ignore
                    pdbseq[code][chain] = line.strip()
        return pdbseq

    def get_nres(self):
        import pandas as pd

        nres = {k: len(v) for k, v in self.seq.items()}
        nres = pd.Series(nres)
        # nres = nres[np.argsort(nres)]
        return nres

    def load_pdb_resl_data(self):
        pdbresl = dict()
        with ipd.open_package_data("pdb/meta/resl.txt.xz") as inp:
            count = 0
            countnoresl = 0
            for line in inp:
                line = line.decode()
                count += 1
                if count < 7:
                    continue
                splt = line.split()
                code = splt[0]
                assert isinstance(code, str)
                if code in pdbresl:
                    log.debug(f"duplicate code {code}")
                assert len(code) == 4
                if len(splt) == 3:
                    resl = float(splt[2])
                else:
                    resl = -1
                    log.debug(f"bad resolu.idx line {line.strip()}")
                if resl == -1:
                    countnoresl += 1
                    resl = 9e9
                pdbresl[code] = resl
        import pandas as pd

        pdbresl = pd.Series(pdbresl)
        # pdbresl = pdbresl[np.argsort(pdbresl)]
        return pdbresl

    def get_compound(self):
        pdbcompound = dict()
        with ipd.open_package_data("pdb/meta/compound.txt.xz") as inp:
            count = 0
            countnoresl = 0
            for line in inp:
                line = line.decode()
                count += 1
                if count < 5:
                    continue
                code = line[:4]
                compound = line[5:].strip()
                # print(code)
                # print(compound)
                # assert 0
                pdbcompound[code] = compound
        return pdbcompound

    def get_clust(self, si):
        clust = list()
        with ipd.open_package_data(f"pdb/meta/clust{si}.txt.xz") as inp:
            for line in inp:
                clust.append(line.split())
        return clust

    def get_biotype(self):
        biotype = dict()
        with ipd.open_package_data("pdb/meta/entries.txt.xz") as inp:
            count = 0
            for line in inp:
                count += 1
                line = line.decode()
                s = line.split("\t")
                if len(s) < 5:
                    continue
                code = s[0]
                bt = s[1]
                # print(code, bt)
                biotype[code] = bt
        return biotype

    def get_source(self):
        foo = dict()
        with ipd.open_package_data("pdb/meta/source.txt.xz") as inp:
            count = 0
            for line in inp:
                count += 1
                if count < 4:
                    continue
                code = line[:4].decode()
                source = line[5:].strip().decode()
                foo[code] = source

        return foo

    def _get_entrytypes_hack(self):
        entrytypes = dict()
        byentrytype = {"prot": set(), "nuc": set(), "prot-nuc": set(), "other": set()}
        with ipd.open_package_data("pdb/meta/entrytypes.txt.xz") as inp:
            for line in inp:
                code, entrytype, _ = line.split()
                code = code.decode().upper()
                entrytype = entrytype.decode()
                entrytypes[code] = entrytype
                byentrytype[entrytype].add(code)
        return entrytypes, byentrytype

    def get_entrytype(self):
        return self._get_entrytypes_hack()[0]

    def get_byentrytype(self):
        return self._get_entrytypes_hack()[1]

    def get_ligcount(self):
        return ipd.load_package_data("pdb/meta/lig/hetres_counts.pickle.xz")

    def get_rescount(self):
        return ipd.load_package_data("pdb/meta/lig/pdb_rescount.pickle.xz")

    def get_ligpdbs(self):
        return ipd.load_package_data("pdb/meta/lig/hetres_pdbs.pickle.xz")

# nifty little, officially approved, hack to use class proterties on 'module'
sys.modules[__name__] = PDBMetadata()  # type: ignore

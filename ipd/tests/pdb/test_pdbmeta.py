import ipd

config_test = ipd.Bunch(
    re_only=[
        #
    ],
    re_exclude=[
        #
    ],
)

def main():
    ipd.tests.maintest(
        namespace=globals(),
        config=config_test,
        verbose=1,
        check_xfail=False,
    )

def test_PDBSearchResult():
    pass

def test_PDBMetadata():
    # PDBMetadata.clear_pickle_cache(self, names)
    # PDBMetadata.get_biotype(self)
    # PDBMetadata.get_byentrytype(self)
    # PDBMetadata.get_clust(self, si)
    # PDBMetadata.get_compound(self)
    # PDBMetadata.get_entrytype(self)
    # PDBMetadata.get_full_seq(self)
    # PDBMetadata.get_ligcount(self)
    # PDBMetadata.get_ligpdbs(self)
    # PDBMetadata.get_nres(self)
    # PDBMetadata.get_rescount(self)
    # PDBMetadata.get_source(self)
    # PDBMetadata.load_pdb_resl_data(self)
    # PDBMetadata.load_pdb_seqres_data(self)
    # PDBMetadata.load_pdb_xtal_data(self)
    # PDBMetadata.make_pdb_set(self, maxresl=2.0, minres=50, maxres=500, max_seq_ident=0.5, pisces_chains=True, entrytype='prot')
    # PDBMetadata.update_source_files(self, replace=True)
    pass

if __name__ == '__main__':
    main()

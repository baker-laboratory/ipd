import ipd

def main():
    ipd.tests.maintest(namespace=globals())

def test_openfile():
    # openfile(fname, mode='r', **kw)
    ipd.dev.openfile(ipd.dev.package_testdata_path('pdb/tiny.pdb')).close()
    ipd.dev.openfile(ipd.dev.package_testdata_path('pdb/1coi.pdb1.gz')).close()

def test_openfile_iterable():
    # openfile(fname, mode='r', **kw)
    files = ipd.dev.openfile([
        ipd.dev.package_testdata_path('pdb/tiny.pdb'),
        ipd.dev.package_testdata_path('pdb/1coi.pdb1.gz'),
    ])
    assert all(not f.closed for f in files)
    [f.close() for f in files]
    assert all(f.closed for f in files)

def test_readfile():
    v1 = ipd.dev.readfile(ipd.dev.package_testdata_path('pdb/tiny.pdb'))
    assert len(v1) == 730

if __name__ == '__main__':
    main()

import gzip
import zipfile
import lzma
from pathlib import Path
from io import BytesIO, StringIO
from ipd.dev.storage.openfile import decompressed_fname, openfile, readfile, istextfile, isbinfile

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

def test_decompressed_fname():
    cases = [
        ("file.txt", "file.txt"),
        ("file.txt.gz", "file.txt"),
        ("file.txt.zip", "file.txt"),
        ("file.txt.xz", "file.txt"),
        ("file.txt.tar", "file.txt"),
        ("/path/to/file.txt", "/path/to/file.txt"),
        ("/path/to/file.txt.gz", "/path/to/file.txt"),
        (Path("file.txt.xz"), Path("file.txt")),
        (Path("/path/to/file.txt.tar"), Path("/path/to/file.txt")),
    ]

    for fname, expected in cases:
        dcomp = decompressed_fname(fname)
        if dcomp != expected:
            ic(type(fname), type(expected))
            print(f'Failed for "{repr(fname)}" expected "{repr(expected)}", got "{repr(dcomp)}"')
        assert dcomp == expected

def test_decompressed_fname_iterable():
    input_files = ["file1.txt.gz", "file2.txt.xz", "file3.txt.zip"]
    expected_outputs = ["file1.txt", "file2.txt", "file3.txt"]
    assert decompressed_fname(input_files) == expected_outputs

def test_openfile_text(tmpdir):
    file_path = tmpdir / "test.txt"
    file_path.write_text("hello world", "utf8")

    with openfile(file_path, mode='r') as f:
        assert f.read() == "hello world"

def test_openfile_gzip(tmpdir):
    file_path = tmpdir / "test.txt.gz"
    with gzip.open(file_path, 'wt') as f:
        f.write("hello gzip")

    with openfile(file_path, mode='rt') as f:
        assert f.read() == "hello gzip"

def test_openfile_xz(tmpdir):
    file_path = tmpdir / "test.txt.xz"
    with lzma.open(file_path, 'wt') as f:
        f.write("hello xz")

    with openfile(file_path, mode='rt') as f:
        assert f.read() == "hello xz"

def test_openfile_zip(tmpdir):
    file_path = tmpdir / "test.zip"
    with zipfile.ZipFile(file_path, 'w') as myzip:
        with myzip.open("my_file.txt", 'w') as f:
            f.write(b"hello zip")

    with openfile(file_path, mode='r') as f:
        assert f.read() == b"hello zip"

def test_readfile_text(tmpdir):
    file_path = tmpdir / "test.txt"
    file_path.write_text("reading test", "utf8")
    assert readfile(file_path) == "reading test"

def test_readfile_iterable(tmpdir):
    file1 = tmpdir / "test1.txt"
    file2 = tmpdir / "test2.txt.gz"
    file1.write_text("file1 content", "utf8")
    with gzip.open(file2, 'wt') as f:
        f.write("file2 content")
    result = readfile([file1, file2])
    assert result == ["file1 content", b"file2 content"]

def test_istextfile():
    f = StringIO("text mode")
    f.mode = "rt"
    assert istextfile(f) is True

def test_isbinfile():
    f = BytesIO(b"binary mode")
    f.mode = "rb"
    assert isbinfile(f) is True

if __name__ == '__main__':
    main()

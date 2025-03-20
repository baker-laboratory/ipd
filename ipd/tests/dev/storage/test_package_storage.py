import pytest

import ipd
import os
import lzma
import tempfile
import pickle

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

from ipd.dev.storage import (
    package_data_path,
    package_testdata_path,
    package_testcif_path,
    load_json,
    dump_json,
    decompress_lzma_file,
    fname_extensions,
    is_pickle_fname,
    load,
    load_pickle,
    save,
    save_pickle,
    is_pdb_fname,
)

def test_package_data_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test.pickle")
        with open(test_file, "w") as f:
            f.write("test")
        assert package_data_path("test", datadir=tmpdir) == test_file

def test_package_testdata_path():
    path = package_testdata_path("testfile")
    assert path.endswith("/data/tests/testfile")

# def test_load_package_data():
# pickle.dump({"key": "value"}, tmpfile)
# tmpfile.flush()
# ipd.icv(tmpfile.name)
# data = load_package_data(tmpfile.name)
# assert data == {"key": "value"}

def test_package_testcif_path():
    path = package_testcif_path("1abc")
    assert path.endswith("tests/pdb/1abc.bcif.gz")

# def test_have_package_data():
# with tempfile.NamedTemporaryFile("wb", suffix=".pickle") as tmpfile:
# assert have_package_data(tmpfile.name) is False
# pickle.dump({"key": "value"}, tmpfile)
# tmpfile.flush()
# assert have_package_data(tmpfile.name) is True

# def test_open_package_data():
# with tempfile.NamedTemporaryFile("w") as tmpfile:
# tmpfile.write("test data")
# tmpfile.flush()
# with open_package_data(tmpfile.name) as f:
# assert f.read() == "test data"

# def test_open_package_file():
# with tempfile.NamedTemporaryFile("w") as tmpfile:
# tmpfile.write("file content")
# tmpfile.flush()
# with open_package_file(tmpfile.name) as f:
# assert f.read() == "file content"

# def test_save_package_data():
# with tempfile.NamedTemporaryFile("wb", suffix=".pickle", delete=False) as tmpfile:
# save_package_data({"key": "value"}, tmpfile.name)
# with open(tmpfile.name, "rb") as f:
# data = pickle.load(f)
# assert data == {"key": "value"}
# os.remove(tmpfile.name)

def test_load_json_and_dump_json():
    with tempfile.NamedTemporaryFile("w", delete=False) as tmpfile:
        dump_json({"key": "value"}, tmpfile.name)
        data = load_json(tmpfile.name)
        assert data == {"key": "value"}
        os.remove(tmpfile.name)

def test_decompress_lzma_file():
    with tempfile.NamedTemporaryFile("wb", suffix=".xz", delete=False) as tmpfile:
        with lzma.open(tmpfile.name, "wb") as f:
            f.write(b"compressed data")
        output_file = tmpfile.name[:-3]

        decompress_lzma_file(tmpfile.name)
        with open(output_file, "rb") as f:
            assert f.read() == b"compressed data"
        os.remove(tmpfile.name)
        os.remove(output_file)

def test_fname_extensions():
    result = fname_extensions("/path/to/file.tar.gz")
    assert result.base == "file"
    assert result.ext == ""
    assert result.compression == ".tar.gz"

def test_is_pickle_fname():
    assert is_pickle_fname("file.pickle") is True
    assert is_pickle_fname("file.json") is False

def test_load():
    with tempfile.NamedTemporaryFile("wb", suffix=".pickle", delete=False) as tmpfile:
        pickle.dump({"key": "value"}, tmpfile)
        tmpfile.close()
        data = load(tmpfile.name)
        assert data == {"key": "value"}
        os.remove(tmpfile.name)

def test_load_pickle():
    with tempfile.NamedTemporaryFile("wb", suffix=".pickle", delete=False) as tmpfile:
        pickle.dump({"key": "value"}, tmpfile)
        tmpfile.close()
        data = load_pickle(tmpfile.name)
        assert data == {"key": "value"}
        os.remove(tmpfile.name)

def test_save():
    with tempfile.NamedTemporaryFile("wb", suffix=".pickle", delete=False) as tmpfile:
        save({"key": "value"}, tmpfile.name)
        with open(tmpfile.name, "rb") as f:
            data = pickle.load(f)
            assert data == {"key": "value"}
        os.remove(tmpfile.name)

def test_save_pickle():
    with tempfile.NamedTemporaryFile("wb", suffix=".pickle", delete=False) as tmpfile:
        save_pickle({"key": "value"}, tmpfile.name)
        with open(tmpfile.name, "rb") as f:
            data = pickle.load(f)
            assert data == {"key": "value"}
        os.remove(tmpfile.name)

def test_is_pdb_fname():
    with tempfile.NamedTemporaryFile(suffix=".pdb") as tmpfile:
        assert is_pdb_fname(tmpfile.name) is True
    with tempfile.NamedTemporaryFile(suffix=".pdb.gz") as tmpfile:
        assert is_pdb_fname(tmpfile.name) is True
    with tempfile.NamedTemporaryFile(suffix=".json") as tmpfile:
        with pytest.raises(ValueError):
            is_pdb_fname(tmpfile.name)

if __name__ == '__main__':
    main()

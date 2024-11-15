import shutil
import subprocess
import threading
import time
from subprocess import check_output

class FileFetcher(threading.Thread):
    def __init__(self, fname, cache):
        super().__init__()
        self.fname = fname
        self.localfname = cache.tolocal(fname)
        self.tmpfname = f'{self.localfname}.tmp'
        self.start()

    def run(self):
        if os.path.exists(self.localfname): return  # type: ignore
        if os.path.exists(self.fname):  # type: ignore
            shutil.copyfile(self.fname, self.tmpfname)
        else:
            out = check_output(f'rsync digs:{self.fname} {self.tmpfname}'.split(), stderr=subprocess.STDOUT)
            print('rsynced')
            if out.lower().count(b'error'): notify(out)  # type: ignore
        shutil.move(self.tmpfname, self.localfname)

class FileCache:
    def __init__(self, fnames, **kw):
        self.fnames = fnames

    def __getitem__(self, i):
        return self.fnames[i]

    def cleanup(self):
        pass

class PrefetchLocalFileCache(FileCache):
    """Copies files to a CONF temp directory.

    Will downloads files ahead of requested index in background.
    """
    def __init__(self, fnames, numprefetch=7, path='/tmp/ppp/filecache'):
        self.path = path
        self.fetchers = {}
        os.makedirs(path, exist_ok=True)  # type: ignore
        self.available = set(os.listdir(path))  # type: ignore
        self.fnames = fnames
        self.numprefetch = numprefetch
        self[0]

    def update_fetchers(self):
        done = {k for k, v in self.fetchers.items() if not v.is_alive()}
        self.available |= done
        for k in done:
            del self.fetchers[k]

    def prefetch(self, fname):
        if isinstance(fname, list):
            assert all(self.prefetch(f) for f in fname)
        if fname in self.available or fname in self.fetchers: return True
        self.update_fetchers()
        if len(self.fetchers) > 10: return False
        self.fetchers[fname] = FileFetcher(fname, self)
        return True

    def tolocal(self, fname):
        slash = '\\'
        return f"{self.path}/{fname.replace('/',slash)}"

    def __getitem__(self, index):
        assert self.prefetch(self.fnames[index])
        for i in range(min(self.numprefetch, len(self.fnames) - index - 1)):
            self.prefetch(self.fnames[index + i + 1])
        localfname = self.tolocal(self.fnames[index])
        for _ in range(100):
            self.update_fetchers()
            if self.fnames[index] in self.available:
                return localfname
            time.sleep(0.1)
        from ipd.dev.qt import isfalse_notify
        isfalse_notify(os.path.exists(localfname))  # type: ignore

    def cleanup(self):
        self.update_fetchers()
        for f in self.fetchers:
            f.killed = True

import os
from pathlib import Path
from subprocess import check_output

import ipd
from ipd import ppp
from ipd.ppp.server.pppapi import PPPBackend

requests = ipd.lazyimport('requests', pip=True)
fastapi = ipd.lazyimport('fastapi', pip=True)
rich = ipd.lazyimport('rich', 'Rich', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
_wpcgit = 'git+https://github.com/willsheffler/wills_pymol_crap'
wills_pymol_crap = ipd.lazyimport('wills_pymol_crap', _wpcgit, pip=True)
pymol = ipd.lazyimport('pymol', 'pymol-bundle', mamba=True, channels='-c schrodinger')
print = rich.print
_GLOBAL_CLIENT = None

def get_hack_fixme_global_client():
    return _GLOBAL_CLIENT

REMOTE_MODE = not os.path.exists('/net/scratch/sheffler')
# profile = ipd.dev.timed
profile = lambda f: f

class PPPClient(ipd.crud.ClientBase, Backend=PPPBackend):
    def __init__(self, server_addr_or_testclient):
        super().__init__(server_addr_or_testclient)
        assert self.get('/')['msg'] == 'Hello World'
        global _GLOBAL_CLIENT
        _GLOBAL_CLIENT = self  #there should be a better way to do this

    def preprocess_get(self, kw):
        return ipd.ppp.fix_label_case(kw)

    def upload_poll(self, poll):
        print('upload_poll')
        poll = poll.to_spec()
        # digs:/home/sheffler/project/rfdsym/hilvert/pymol_saves
        if digs := poll.path.startswith('digs:'):
            lines = check_output(['rsync', f'{poll.path}/*']).decode().splitlines()
            poll.path = poll.path[5:]
            fnames = [os.path.join(poll.path, l.split()[4]) for l in lines]
        else:
            assert os.path.isdir(poll.path)
            fnames = [os.path.join(poll.path, f) for f in os.listdir(poll.path)]
        filt = lambda s: not s.startswith('_') and s.endswith(ipd.STRUCTURE_FILE_SUFFIX) and os.stat(s).st_size
        fnames = list(filter(filt, fnames))
        assert fnames, f'path must contain structure files: {poll.path}'
        poll = self.upload(poll, _dispatch_on_type=False)
        if ipd.dev.qt.isfalse_notify(not isinstance(poll, str), poll): return
        construct = ppp.PollFileSpec.construct if digs else ppp.PollFileSpec
        files = [construct(pollid=poll.id, fname=fn) for fn in fnames]
        if result := self.post('/create/pollfiles', files):
            self.remove(poll)
            assert 0, result
        return poll

    def upload_review(self, review):
        review = review.to_spec()
        file = self.pollfile(pollid=review.pollid, id=review.pollfileid)
        exists, permafname = self.get('/have/pollfile', fname=file.fname, pollid=file.pollid)
        file.permafname = permafname
        assert file.permafname
        file, fileid = file.to_spec(), file.id
        assert self.pollfile(id=fileid).permafname == permafname
        file.filecontent = Path(file.fname).read_text()
        assert file.filecontent
        if not exists:
            if response := self.post('/create/pollfilecontents', file): return response
        review = review.to_spec()
        return self.upload(review, _dispatch_on_type=False)

    def pollinfo(self, user=None):
        return self.get(f'/pollinfo?user={user}')

    def pymolcmdsdict(self):
        return self.get('/pymolcmds')

    def poll_fids(self, id):
        return self.get(f'/poll{id}/fids')

    # def create_poll(self, poll):
    # self.post('/poll', json=json.loads(poll.model_dump_json()))
    def reviews_for_fname(self, fname):
        fname = fname.replace('/', '__DIRSEP__')
        rev = self.get(f'/reviews/byfname/{fname}')
        return [pppp.Review(self, **_) for _ in rev]

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# the backend models come from here
# class DBProll
# class DBFileKind
# class DBPollFile
# class DBReview
# class DBReviewStep
# class DBPymolCMD
# class DBWorkflow
# class DBFlowStep
# class DBUser
# class DBGroup
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
for cls in PPPClient.__client_models__.values():
    globals()[cls.__name__] = cls

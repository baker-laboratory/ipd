import os
from datetime import datetime
import functools
import tempfile
import contextlib
from subprocess import check_output
import ipd
from pathlib import Path
import gzip
import getpass
import traceback
import socket
from typing import Optional, Union
from ipd.sym.guess_symmetry import guess_symmetry, guess_sym_from_directory

import pydantic

requests = ipd.lazyimport('requests', pip=True)
rich = ipd.lazyimport('rich', 'Rich', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
wills_pymol_crap = ipd.lazyimport('wills_pymol_crap',
                                  'git+https://github.com/willsheffler/wills_pymol_crap',
                                  pip=True)
pymol = ipd.lazyimport('pymol', 'pymol-bundle', mamba=True, channels='-c schrodinger')
print = rich.print

from ipd.ppp.clientmodels import Poll, Review, ReviewStep, File, PymolCMD, FlowStep, Workflow, User, Group

REMOTE_MODE = not os.path.exists('/net/scratch/sheffler')
# profile = ipd.dev.timed
profile = lambda f: f

class PPPClientError(Exception):
    pass

def tojson(thing):
    if isinstance(thing, list):
        return f'[{",".join(tojson(_) for _ in thing)}]'
    return thing.json()

class PPPClient:
    def __init__(self, server_addr_or_testclient):
        if isinstance(server_addr_or_testclient, str):
            print('PPPClient: connecting to server', server_addr_or_testclient)
            self.testclient, self.server_addr = None, server_addr_or_testclient
        else:
            print('PPPClient: using testclient')
            self.testclient = server_addr_or_testclient
        assert self.get('/')['msg'] == 'Hello World'

    def getattr(self, thing, id, attr):
        return self.get(f'/getattr/{thing}/{id}/{attr}')

    def get(self, url, **kw):
        ipd.ppp.fix_label_case(kw)
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        if not self.testclient: url = f'http://{self.server_addr}/ppp{url}'
        if self.testclient: response = self.testclient.get(url)
        else: response = requests.get(url)
        if response.status_code != 200:
            reason = response.reason if hasattr(response, 'reason') else '???'
            raise PPPClientError(f'GET failed URL: "{url}"\n    RESPONSE: {response}\n    '
                                 f'REASON:   {reason}\n    CONTENT:  {response.content.decode()}')
        return response.json()

    def post(self, url, thing, **kw):
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        if not self.testclient: url = f'http://{self.server_addr}/ppp{url}'
        # print('POST', url, thing)
        body = tojson(thing)
        if self.testclient: response = self.testclient.post(url, content=body)
        else: response = requests.post(url, body)
        if response.status_code != 200:
            if len(str(body)) > 512: body = f'{body[:200]} ... {body[-200:]}'
            reason = response.reason if hasattr(response, 'reason') else '???'
            raise PPPClientError(f'POST failed "{url}"\n    BODY:     {body}\n    '
                                 f'RESPONSE: {response}\n    REASON:   {reason}\n    '
                                 f'CONTENT:  {response.content.decode()}')
        return response.json()

    def remove(self, thing):
        if isinstance(thing, Poll): return self.get(f'/remove/poll/{thing.id}')
        elif isinstance(thing, File): return self.get(f'/remove/file/{thing.id}')
        elif isinstance(thing, Review): return self.get(f'/remove/review/{thing.id}')
        elif isinstance(thing, PymolCMD): return self.get(f'/remove/pymolcmd/{thing.id}')
        else: raise ValueError('cant remove type {type(thing)}\n{thing}')

    def upload(self, thing, **kw):
        # print('upload', type(thing), kw)
        if thing._errors: return thing._errors
        kind = type(thing).__name__.replace('Spec', '').lower()
        return self.post(f'/create/{kind}', thing, **kw)

    def upload_poll(self, poll):
        # digs:/home/sheffler/project/rfdsym/hilvert/pymol_saves
        if digs := poll.path.startswith('digs:'):
            lines = check_output(['rsync', f'{poll.path}/*']).decode().splitlines()
            poll.path = poll.path[5:]
            fnames = [os.path.join(poll.path, l.split()[4]) for l in lines]
        else:
            assert os.path.isdir(poll.path)
            fnames = [os.path.join(poll.path, f) for f in os.listdir(poll.path)]
        filt = lambda s: not s.startswith('_') and s.endswith(ipd.ppp.STRUCTURE_FILE_SUFFIX)
        fnames = list(filter(filt, fnames))
        assert fnames, f'path must contain structure files: {poll.path}'
        if errors := self.upload(poll): return errors
        poll = self.polls(name=poll.name)[0]
        construct = FileSpec.construct if digs else FileSpec
        files = [construct(pollid=poll.id, fname=fn, user=getpass.getuser()) for fn in fnames]
        return self.post('/create/files', files)

    def upload_review(self, review, fname=None):
        fname = fname or review.fname
        file = FileSpec(pollid=review.pollid, fname=fname)
        print('review.fname', review.pollid, fname)
        exists, permafname = self.post('/have/file', file)
        review.permafname = permafname
        file.permafname = permafname
        file.filecontent = Path(fname).read_text()
        if not exists:
            if response := self.post('/create/file', file): return response
        review = ReviewSpec(**review.dict())
        return self.upload(review)

    def pollinfo(self, user=None):
        print(f'client pollinfo {user}')
        if self.testclient: return self.testclient.get(f'/pollinfo?user={user}').json()
        response = requests.get(f'http://{self.server_addr}/pollinfo?user={user}')
        if response.content: return response.json()
        return []

    def polls(self, **kw):
        return [Poll(self, **p) for p in self.get('/polls', **kw)]

    def reviews(self, **kw):
        return [Review(self, **_) for _ in self.get('/reviews', **kw)]

    def files(self, **kw):
        return [File(self, **_) for _ in self.get('/files', **kw)]

    def pymolcmds(self, **kw):
        return [PymolCMD(self, **_) for _ in self.get('/pymolcmds', **kw)]

    def pymolcmdsdict(self):
        return self.get('/pymolcmds')

    def poll(self, id):
        return Poll(self, **self.get(f'/poll{id}'))

    def pymolcmd(self, id):
        return PymolCMD(self, **self.get(f'/pymolcmd{id}'))

    def poll_fids(self, id):
        return self.get(f'/poll{id}/fids')

    # def create_poll(self, poll):
    # self.post('/poll', json=json.loads(poll.json()))

    def reviews_for_fname(self, fname):
        fname = fname.replace('/', '__DIRSEP__')
        rev = self.get(f'/reviews/byfname/{fname}')
        return [Review(self, **_) for _ in rev]

    def _add_some_cmds(self):
        self.upload(
            PymolCMDSpec(
                name='sym: Make {sym.upper()}',
                cmdstart='from wills_pymol_crap import symgen',
                cmdon=
                f'symgen.make{sym}("$subject", name="sym"); delete $subject; cmd.set_name("sym", "$subject")',
                cmdoff='remove not chain A',
            ))

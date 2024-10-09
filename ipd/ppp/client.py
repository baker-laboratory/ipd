import os
from subprocess import check_output
import ipd
import functools
from ipd import ppp
from pathlib import Path
import getpass
from typing import Union

requests = ipd.lazyimport('requests', pip=True)
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

clientmodels = dict(
    poll=ppp.Poll,
    review=ppp.Review,
    reviewstep=ppp.ReviewStep,
    pollfile=ppp.PollFile,
    pymolcmd=ppp.PymolCMD,
    flowstep=ppp.FlowStep,
    workflow=ppp.Workflow,
    user=ppp.User,
    group=ppp.Group,
)
assert not any(name.endswith('s') for name in clientmodels)

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
        global _GLOBAL_CLIENT
        _GLOBAL_CLIENT = self  #there should be a better way to do this

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
        if isinstance(thing, ppp.Poll): return self.get(f'/remove/poll/{thing.id}')
        elif isinstance(thing, ppp.PollFile): return self.get(f'/remove/pollfile/{thing.id}')
        elif isinstance(thing, pppp.Review): return self.get(f'/remove/review/{thing.id}')
        elif isinstance(thing, ppp.PymolCMD): return self.get(f'/remove/pymolcmd/{thing.id}')
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
        construct = ppp.PollFileSpec.construct if digs else ppp.PollFileSpec
        files = [construct(pollid=poll.id, fname=fn, user=getpass.getuser()) for fn in fnames]
        return self.post('/create/pollfiles', files)

    def upload_review(self, review, fname=None):
        fname = fname or review.fname
        file = ppp.PollFileSpec(pollid=review.pollid, fname=fname)
        print('review.fname', review.pollid, fname)
        exists, permafname = self.post('/have/pollfile', file)
        review.permafname = permafname
        file.permafname = permafname
        file.pollfilecontent = Path(fname).read_text()
        if not exists:
            if response := self.post('/create/pollfile', file): return response
        review = ppp.ReviewSpec(**review.dict())
        return self.upload(review)

    def pollinfo(self, user=None):
        print(f'client pollinfo {user}')
        if self.testclient: return self.testclient.get(f'/pollinfo?user={user}').json()
        response = requests.get(f'http://{self.server_addr}/pollinfo?user={user}')
        if response.content: return response.json()
        return []

    def pymolcmdsdict(self):
        return self.get('/pymolcmds')

    def poll_fids(self, id):
        return self.get(f'/poll{id}/fids')

    # def create_poll(self, poll):
    # self.post('/poll', json=json.loads(poll.json()))

    def reviews_for_fname(self, fname):
        fname = fname.replace('/', '__DIRSEP__')
        rev = self.get(f'/reviews/byfname/{fname}')
        return [pppp.Review(self, **_) for _ in rev]

    def _add_some_cmds(self):
        self.upload(
            ppp.PymolCMDSpec(
                name='sym: Make {sym.upper()}',
                cmdstart='from wills_pymol_crap import symgen',
                cmdon=
                f'symgen.make{sym}("$subject", name="sym"); delete $subject; cmd.set_name("sym", "$subject")',
                cmdoff='remove not chain A',
            ))

# Generic interface for accessing models from the server. Any name or name suffixed with 's'
# that is in clientmodels, above, will get /name from the server and turn the result(s) into
# the appropriate client model type, list of such types for plural, or None.

for _name, _cls in clientmodels.items():

    def make_funcs_forcing_closure_over_cls_name(cls=_cls, name=_name):
        def multi(self, **kw) -> list[cls]:
            return [cls(self, **x) for x in self.get(f'/{name}s', **kw)]

        def single(self, **kw) -> Union[cls, None]:
            result = self.get(f'/{name}', **kw)
            return cls(self, **result) if result else None

        return single, multi

    single, multi = make_funcs_forcing_closure_over_cls_name()
    setattr(PPPClient, _name, single)
    setattr(PPPClient, f'{_name}s', multi)

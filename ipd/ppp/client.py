import os
from subprocess import check_output
import ipd
import functools
from ipd import ppp
from pathlib import Path
import getpass
from typing import Union

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

class PPPClientError(Exception):
    pass

def tojson(thing):
    if isinstance(thing, list):
        return f'[{",".join(tojson(_) for _ in thing)}]'
    if isinstance(thing, str):
        return thing
    return thing.json()

class PPPClient:
    def __init__(self, server_addr_or_testclient):
        if isinstance(server_addr_or_testclient, str):
            print('PPPClient: connecting to server', server_addr_or_testclient)
            self.testclient, self.server_addr = None, server_addr_or_testclient
        elif isinstance(fastapi.testclient.TestClient):
            print('PPPClient: using testclient')
            self.testclient = server_addr_or_testclient
        assert self.get('/')['msg'] == 'Hello World'
        global _GLOBAL_CLIENT
        _GLOBAL_CLIENT = self  #there should be a better way to do this

    def getattr(self, thing, id, attr):
        return self.get(f'/getattr/{thing}/{id}/{attr}')

    def setattr(self, thing, attr, val):
        thingtype = thing.__class__.__name__.lower()
        return self.post(f'/setattr/{thingtype}/{thing.id}/{attr}', val)

    def get(self, url, **kw):
        ipd.ppp.fix_label_case(kw)
        query = '&'.join([f'{k}={v}' for k, v in kw.items()])
        url = f'{url}?{query}' if query else url
        if not self.testclient: url = f'http://{self.server_addr}/ppp{url}'
        if self.testclient:
            return self.testclient.get(url)
        response = requests.get(url)
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
            if len(str(body)) > 2048: body = f'{body[:1024]} ... {body[-1024:]}'
            reason = response.reason if hasattr(response, 'reason') else '???'
            raise PPPClientError(f'POST failed "{url}"\n    BODY:     {body}\n    '
                                 f'RESPONSE: {response}\n    REASON:   {reason}\n    '
                                 f'CONTENT:  {response.content.decode()}')
        return response.json()

    def remove(self, thing):
        assert isinstance(thing, ipd.ppp.SpecBase), f'cant remove type {thing.__class__.__name__}'
        thingname = thing.__class__.__name__.replace('Spec', '').lower()
        return self.get(f'/remove/{thingname}/{thing.id}')

    def upload(self, thing, _custom=True, **kw):
        if _custom and isinstance(thing, ipd.ppp.PollSpec): return self.upload_poll(thing)
        if _custom and isinstance(thing, ipd.ppp.ReviewSpec): return self.upload_review(thing)
        thing = thing.spec()
        # print('upload', type(thing), kw)
        if thing._errors: return thing._errors
        kind = type(thing).__name__.replace('Spec', '').lower()
        result = self.post(f'/create/{kind}', thing, **kw)
        if not result.isdigit(): return result
        return ipd.ppp.frontend_model[kind](self, **self.get(f'/{kind}', id=result))

    def upload_poll(self, poll):
        poll = poll.spec()
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
        poll = self.upload(poll, _custom=False)
        construct = ppp.PollFileSpec.construct if digs else ppp.PollFileSpec
        files = [construct(pollid=poll.id, fname=fn) for fn in fnames]
        return self.post('/create/pollfiles', files)

    def upload_review(self, review):
        print('=================================================================')
        review = review.spec()
        file = self.pollfile(pollid=review.pollid, id=review.pollfileid)
        print('review fname', review.pollid, file.fname)
        exists, permafname = self.get('/have/pollfile', fname=file.fname, pollid=file.pollid)
        assert permafname
        file.permafname = permafname
        file, fileid = file.spec(), file.id
        assert self.pollfile(id=fileid).permafname == permafname
        print('id/perma', fileid, permafname)
        file.filecontent = Path(file.fname).read_text()
        if not exists:
            if response := self.post('/create/pollfilecontents', file): return response
        review = ppp.ReviewSpec(**review.dict())
        return self.upload(review, _custom=False)

    def pollinfo(self, user=None):
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

# Generic interface for accessing models from the server. Any name or name suffixed with 's'
# that is in frontend_model, above, will get /name from the server and turn the result(s) into
# the appropriate client model type, list of such types for plural, or None.

for _name, _cls in ipd.ppp.frontend_model.items():

    def make_funcs_forcing_closure_over_cls_name(cls=_cls, name=_name):
        def new(self, **kw) -> str:
            return self.upload(ipd.ppp.spec_model[name](**kw))

        def count(self, **kw) -> int:
            return self.get(f'/n{name}s', **kw)

        def multi(self, **kw) -> list[cls]:
            return [cls(self, **x) for x in self.get(f'/{name}s', **kw)]

        def single(self, **kw) -> Union[cls, None]:
            result = self.get(f'/{name}', **kw)
            return cls(self, **result) if result else None

        return single, multi, count, new

    single, multi, count, new = make_funcs_forcing_closure_over_cls_name()
    setattr(PPPClient, _name, single)
    setattr(PPPClient, f'{_name}s', multi)
    setattr(PPPClient, f'n{_name}s', count)
    setattr(PPPClient, f'new{_name}', new)

import os
from subprocess import check_output
import ipd
from ipd import ppp
import uuid
from pathlib import Path

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

class PPPClient(ipd.dev.ModelFrontend, models=ipd.ppp.client_model):
    def __init__(self, server_addr_or_testclient):
        if isinstance(server_addr_or_testclient, str):
            # print('PPPClient: connecting to server', server_addr_or_testclient)
            self.testclient, self.server_addr = None, server_addr_or_testclient
        elif isinstance(fastapi.testclient.TestClient):
            # print('PPPClient: using testclient')
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
        body = tojson(thing)
        print('POST', url, type(thing))
        if self.testclient: response = self.testclient.post(url, content=body)
        else: response = requests.post(url, body)
        ic(response)
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

    def upload(self, thing, _dispatch_on_type=True, **kw):
        if _dispatch_on_type and isinstance(thing, ipd.ppp.PollSpec): return self.upload_poll(thing)
        if _dispatch_on_type and isinstance(thing, ipd.ppp.ReviewSpec): return self.upload_review(thing)
        thing = thing.to_spec()
        # print('upload', type(thing), kw)
        if thing._errors:
            return thing._errors
        kind = type(thing).__name__.replace('Spec', '').lower()
        ic(kind)
        result = self.post(f'/create/{kind}', thing, **kw)
        ic(result)
        try:
            result = uuid.UUID(result)
            return ipd.ppp.client_model[kind](self, **self.get(f'/{kind}', id=result))
        except ValueError:
            return result

    def upload_poll(self, poll):
        poll = poll.to_spec()
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
        poll = self.upload(poll, _dispatch_on_type=False)
        if ipd.qt.isfalse_notify(not isinstance(poll, str), poll): return
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
        # print(self.user(id=review.userid))
        # print(self.poll(id=review.pollid))
        # print(self.pollfile(id=review.pollfileid))
        # print(self.workflow(id=review.workflowid))
        # assert 0
        result = self.upload(review, _dispatch_on_type=False)
        return result

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

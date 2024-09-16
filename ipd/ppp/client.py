import os
import pydantic
import datetime
import functools
import json
from ordered_set import OrderedSet as ordset
import ipd
from typing import Any, Optional

class PPPClientError(Exception):
    pass

class PollUpload(pydantic.BaseModel):
    pollid: int | None = None
    name: str = ''
    desc: str = ''
    path: str
    public: bool = False
    telemetry: bool = False
    start: datetime.datetime = datetime.datetime.now()
    end: datetime.datetime = datetime.datetime.now()
    props: list[str] = []
    attrs: dict[str, str | int] = {}

    @pydantic.model_validator(mode='after')
    def _update(self):
        if not self.name: self.name = os.path.basename(self.path)
        self.name = self.name.title()
        if not self.desc: self.desc = f"PDBs in {self.path}"
        return self

    @pydantic.validator('props')
    def _valprops(cls, v):
        return list(v)

    def __hash__(self):
        return hash(self.path)

class ClientPoll(PollUpload):
    _client: Optional['PPPClient'] = None
    fids: dict[str, int] | None = None

    def __init__(self, client, **kw):
        super().__init__(**kw)
        self._client = client
        self.fids = client.poll_fids(self.pollid)

class ReviewUpload(pydantic.BaseModel):
    reviewid: int | None = None
    pollid: int
    fname: str
    grade: str
    user: str = ''
    comment: str = ''
    seconds: int = -1
    attrs: dict[str, str | int | float] = {}

    @pydantic.validator('grade')
    def valgrade(cls, grade):
        assert grade.upper() in 'SABCDF'
        return grade.upper()

    @pydantic.validator('fname')
    def valfname(cls, fname):
        assert os.path.exists(fname)
        return os.path.abspath(fname)

class ClientReview(ReviewUpload):
    _client: Optional['PPPClient'] = None
    fids: dict[str, int] | None = None

    def __init__(self, client, **kw):
        super().__init__(**kw)
        self._client = client

class PPPClient:
    def __init__(self, server_addr_or_testclient):
        if isinstance(server_addr_or_testclient, str): self.server_addr = server_addr_or_testclient
        else: self.testclient = server_addr_or_testclient
        assert self._get('/')['msg'] == 'Hello World'

    def _get(self, url):
        if self.testclient: response = self.testclient.get(url)
        else: response = requests.get(f'http://{self.server_addr}/ppp{url}')
        if response.status_code != 200: raise PPPClientError(f'GET failed {url}')
        return response.json()

    def _post(self, url, **kw):
        if self.testclient: response = self.testclient.post(url, **kw)
        else: requests.post(f'http://{self.server_addr}/ppp{url}', **kw)
        if response.status_code != 200: raise PPPClientError(f'POST failed {url} {kw}')
        return response.json()

    def post(self, thing):
        if isinstance(thing, PollUpload): return self.create_poll(thing)
        else: raise ValueError(f'dont know how to post {type(thing)}')

    def polls(self):
        return {p['name']: ClientPoll(self, **p) for p in self._get('/polls').json()}

    def poll(self, pollid):
        return ClientPoll(self, **self._get(f'/poll{pollid}'))

    def poll_fids(self, pollid):
        return self._get(f'/poll{pollid}/fids')

    def create_poll(self, poll):
        self._post('/poll', json=json.loads(poll.json()))

    def reviews(self):
        return [ClientReview(self, **_) for _ in self._get('/reviews')]

    def reviews_for_pollid(self, pollid):
        return [ClientReview(self, **_) for _ in self._get(f'/reviews/poll{pollid}')]

    def reviews_for_fileid(self, fileid):
        return [ClientReview(self, **_) for _ in self._get(f'/reviews/file{fileid}')]

    def reviews_for_fname(self, fname):
        fname = fname.replace('/', '__DIRSEP__')
        rev = self._get(f'/reviews/byfname/{fname}')
        return [ClientReview(self, **_) for _ in rev]

    def post_review(self, review: ReviewUpload):
        # print('POST REVIEW', review)
        self._post(f'/poll{review.pollid}/review', json=json.loads(review.json()))

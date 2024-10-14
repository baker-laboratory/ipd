import sqlalchemy
import itertools as it
import sys
import ipd
from ipd import ppp
import tempfile
from pathlib import Path
import traceback
import os
import inspect
import subprocess
from fastapi.testclient import TestClient
from sqlmodel import create_engine
from icecream import ic
import pydantic
import pytest
import rich
# from rich import print

def set_debug_requests():
    import requests
    import logging
    import http.client as http_client

    http_client.HTTPConnection.debuglevel = 1
    # You must initialize logging, otherwise you'll not see debug output.
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

def main():
    for fn in [f for n, f in globals().items() if n.startswith('test_')]:
        with tempfile.TemporaryDirectory() as tempdir:
            print('=' * 20, fn, '=' * 20)
            server, backend, client, testclient = make_tmp_clent_server(tempdir)
            try:
                args = {p: locals()[p] for p in inspect.signature(fn).parameters}
                fn(**args)
            except pydantic.ValidationError as e:
                print(e)
                rich.print(e.errors())
                print(traceback.format_exc())
                server.stop()
                return False
            server.stop()
    print('PASS')
    ipd.dev.global_timer.report()

def make_tmp_clent_server(tempdir):
    # engine = create_engine(f'sqlite:///{tempdir}/test.db')
    # backend = ppp.server.Backend(engine, tempdir)
    # server, backend = ipd.ppp.server.run(port=12345, dburl='postgresql://localhost/ppp')
    assert not os.path.exists(f'{tempdir}/test.db')
    server, backend, client = ipd.ppp.server.run(port=12346,
                                                 dburl=f'sqlite:///{tempdir}/test.db',
                                                 workers=1,
                                                 loglevel='warning')
    testclient = TestClient(backend.app)
    # ppp.server.defaults.ensure_init_db(backend)
    return server, backend, client, testclient

def test_poll_attr(client):
    poll = client.upload_poll(ppp.PollSpec(name='foo', path='.'))
    poll.print_full()
    assert all([poll.id == p.pollid for p in poll.pollfiles])
    assert isinstance(poll.pollfiles[0], ppp.PollFile)

def test_spec_srict_ctor_override():
    class Foo(pydantic.BaseModel, ipd.ppp.StrictFields):
        bar: int

    a = Foo(bar=6)
    with pytest.raises(TypeError):
        b = Foo(baz=6)

    class FooNonstrict(Foo):
        bar: int

        def __init__(self, zaz, **kw):
            super().__init__(**kw)

    c = FooNonstrict(zaz=6, bar=6)

def test_setattr(client):
    user = client.newuser(name='foo', fullname='bar')
    assert user.fullname == 'bar'
    user.fullname = 'baz'
    usercopy = client.user(id=user.id)
    assert usercopy.fullname == 'baz'
    assert user.fullname == 'baz'

def test_pollinfo(client, backend):
    binfo = backend.pollinfo()
    info = client.pollinfo()
    assert binfo == info

def test_pymolcmdsdict(client):
    pcd = client.pymolcmdsdict()
    assert isinstance(pcd, list)
    assert isinstance(pcd[0], dict)
    cmds = client.pymolcmds()
    cmd = cmds[0]

def test_ghost(backend):
    user = backend.newuser(name='jameswoods')
    poll = backend.newpoll(name='foo', path='bar', user=user)
    file = backend.newpollfile(fname='baz', poll=poll)
    poll.pollfiles.append(file)
    backend.session.commit()
    assert not file.ghost
    poll.clear(backend)
    assert file.ghost
    assert 1 == len(backend.pollinfo(user=user.name))
    assert not poll.ghost
    backend.remove('poll', poll.id)
    assert poll.ghost
    assert 0 == len(backend.pollinfo(user=user.name))

def test_user_backend(backend):
    foo = backend.newuser(name='foo')
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        backend.newuser(name='foo')
    backend.session.rollback()
    follower = backend.newuser(name='following1')
    follower.following.append(foo)
    follower.following.append(foo)
    follower.following.append(foo)
    backend.session.commit()
    assert follower in foo.followers
    assert len(foo.followers) == 1
    assert len(follower.following) == 1
    foo.followers = []
    assert follower not in foo.followers
    assert foo not in follower.following

def test_spec_basics():
    with pytest.raises(TypeError):
        ppp.PollSpec(name='foo', path='.', userid='test', ntisearien=1)

def test_access_all(client, backend):
    assert 0 == (len(client.polls()))

def test_read_root(client):
    assert client.get("/") == {"msg": "Hello World"}

def _test_file_upload(client, backend):
    # client = ppp.PPPClient(testclient)
    path = ipd.testpath('ppppdbdir')
    spec = ppp.PollSpec(name='usertest1pub', path=path, userid='test', ispublic=True)
    if response := client.upload_poll(spec): print(response)
    localfname = os.path.join(path, '1pgx.cif')
    file = ipd.ppp.PollFileSpec(pollid=spec.id, fname=localfname)
    poll = client.upload(ipd.ppp.PollSpec(name='foo', path='.'))
    assert isinstance(poll, ipd.ppp.Poll), poll
    exists, newfname = client.get('/have/pollfile', fname=file.fname, pollid=client.polls()[-1].id)
    assert file.fname in [f.fname for f in client.pollfiles()]
    assert newfname.endswith('\\ipd\\tests\\data\\ppppdbdir\\1pgx.cif')
    exists, newfname = backend.have_pollfile(fname=localfname, pollid=client.polls()[-1].id)
    # print(exists, f'"{newfname}"')
    assert newfname.endswith('\\ipd\\tests\\data\\ppppdbdir\\1pgx.cif')
    filecontent = Path(localfname).read_text()
    file.filecontent = filecontent
    file.permafname = newfname
    file = ipd.ppp.PollFileSpec(**file.dict())
    # response = backend.create_file(file)
    client.post('/create/pollfilecontents', file)
    print(newfname)
    diff = subprocess.check_output(['diff', localfname, newfname])
    if diff: print(f'diff {localfname} {newfname} {diff}')
    assert not diff
    os.remove(newfname)
    file.filecontent = ''
    files = [file._copy_with_newid() for _ in range(10)]
    backend.create_empty_files(files)
    poll = client.polls(name='usertest1pub')[0]
    print(client.npollfiles())
    # for p in client.pollfiles():
    # print(f'{p.poll.name} {p.fname}')
    assert client.npollfiles() == 15
    assert len(client.pollfiles()) == 15
    client.remove(poll)
    assert len(client.polls()) == 1

def test_poll(client, backend):
    path = ipd.testpath('ppppdbdir')
    client.upload(ppp.UserSpec(name='test1'))

    assert client.user(name='test1').fullname == ''
    client.user(name='test1').fullname = 'fulllname'
    assert client.user(name='test1').fullname == 'fulllname'

    client.upload(ppp.UserSpec(name='test2'))
    client.upload(ppp.UserSpec(name='test3'))
    client.upload_poll(ppp.PollSpec(name='usertest1pub', path=path, userid='test1', ispublic=True))
    client.upload_poll(ppp.PollSpec(name='usertest1pri', path=path, userid='test1', ispublic=False))
    client.upload_poll(ppp.PollSpec(name='usertest2pub', path=path, userid='test2', ispublic=True))
    client.upload_poll(ppp.PollSpec(name='usertest3pri', path=path, userid='test3', ispublic=False))
    assert 3 == len(client.pollinfo(user='test1'))
    assert 2 == len(client.pollinfo(user='test2'))
    assert 3 == len(client.pollinfo(user='test3'))
    assert 2 == len(client.pollinfo(user='sheffler'))
    assert 5 == len(client.pollinfo(user='admin'))

    poll = client.upload_poll(ppp.PollSpec(name='foo1', desc='bar', path=path))
    assert isinstance(poll, ppp.Poll)
    poll = client.upload_poll(ppp.PollSpec(name='foo2', desc='Nntsebar', path=path))
    assert isinstance(poll, ppp.Poll)
    poll = client.upload_poll(
        ppp.PollSpec(name='foo3', desc='barntes', path=path, props=['ligand', 'multichain']))

    assert len(client.polls()) == 7
    polljs = client.get('/polls')
    # print(polljs)
    # polls = [ppp.Poll(**_) for _ in polljs]
    polls = backend.polls()
    # for poll in polls:
    # print(poll, len(poll.files))
    # print(list(poll.files)[:2])

    pfiles = polls[1].pollfiles
    assert len(pfiles) == 4

    poll3 = client.polls()[6]
    # print(poll3)
    assert 'ligand' in poll3.props

    poll = ppp.PollSpec(name='foobar', path=path, props=['ast'])
    # print(poll)
    poll = client.upload_poll(poll)
    assert len(client.polls()) == 8
    poll = client.polls()[6]
    assert isinstance(poll, ipd.ppp.Poll)
    # for p in client.polls():
    # print(p.id, p.name, len(p.files))
    assert len(poll.pollfiles) == 4
    assert isinstance(poll.pollfiles[0], ipd.ppp.PollFile)

    fname = ipd.testpath('ppppdbdir/1pgx.cif')
    client.newuser(name='reviewer')
    print([p.name for p in client.users()])
    # print(client.user(name='reviewer'))
    ic()
    rev = client.newreview(userid='reviewer',
                           pollid=poll.id,
                           pollfileid=poll.pollfiles[1],
                           workflowid='Manual',
                           grade='dislike')
    ic()
    assert client.reviews()[0].user.name == 'reviewer'
    assert os.path.exists(rev.pollfile.permafname)
    # print('\n'.join([f'{f.pollid} {f.fname}' for f in client.pollfiles()]))
    ic()
    review = ppp.ReviewSpec(pollid=2, pollfileid=fname, grade='superlike', comment='foobar')
    assert review.grade == 'superlike'
    result = client.upload_review(review)
    assert isinstance(result, ipd.ppp.Review)
    # rich.print(client.reviews())
    assert len(client.reviews()) == 2
    result = client.upload(ppp.ReviewSpec(pollid=3, pollfileid=fname, grade='hate'))
    assert isinstance(result, ipd.ppp.Review)
    ic()
    assert 1 == len(client.pollfile(id=poll.id, fname=fname).reviews)
    assert 3 == len(list(it.chain(*(f.reviews for f in client.pollfiles(fname=fname)))))
    reviews = client.reviews()
    polls = client.polls()
    files = client.pollfiles()

    assert reviews[0].poll.id == 8
    assert reviews[0].pollfile.id == 28
    for i, p in enumerate(client.polls()):
        print(i, len(p.reviews))
    assert isinstance(files[0], ppp.PollFile)
    assert isinstance(polls[2].pollfiles[0], ppp.PollFile)
    assert isinstance(reviews[2].pollfile, ppp.PollFile)
    assert isinstance(reviews[0], ppp.Review)
    assert isinstance(polls[1].reviews[0], ppp.Review)
    assert isinstance(files[1].poll, ppp.Poll)

    for r in reviews:
        # print('permafname', r.pollfile.permafname)
        # print(r.pollfile)
        assert os.path.exists(r.pollfile.permafname)

    # print([p.name for p in client.polls()])
    # print(len(client.polls(name='foo1')))

    assert len(client.polls(name='foo1')) == 1

    result = client.upload(
        ppp.PymolCMDSpec(name='test', cmdon='show lines', cmdoff='hide lines', userid='test'))
    assert isinstance(result, ppp.PymolCMD)
    result = client.upload(ppp.PymolCMDSpec(name='test2', cmdon='fubar', cmdoff='hide lines',
                                            userid='test')).count('NameError')
    # print(result)
    result = client.upload(
        ppp.PymolCMDSpec(name='test', cmdon='show lines', cmdoff='hide lines', userid='test'))
    # print(result)

    assert len(client.pymolcmds(user='test')) == 1

if __name__ == '__main__':
    main()

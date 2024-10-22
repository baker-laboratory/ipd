import pytest

pytest.importorskip('pymol')
pytest.importorskip('sqlmodel')
pytest.importorskip('fastapi')

import inspect
import itertools as it
import os
import subprocess
import traceback
from pathlib import Path

import pydantic
import pytest
import rich
from fastapi.testclient import TestClient

import ipd
from ipd import ppp

# from rich import print

def set_debug_requests():
    import http.client as http_client
    import logging

    http_client.HTTPConnection.debuglevel = 1
    # You must initialize logging, otherwise you'll not see debug output.
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.DEBUG)
    requests_log.propagate = True

def main():
    from ipd.tests.ppp.conftest import ppp_test_stuff
    with ppp_test_stuff() as (pppbackend, pppserver, pppclient, ppptestclient):
        # pppserver, pppbackend, pppclient, ppptestclient = make_tmp_clent_server()
        for fn in [f for n, f in globals().items() if n.startswith('test_')]:
            pppbackend._clear_all_data_for_testing_only()
            ipd.ppp.server.defaults.add_defaults()
            print('=' * 20, fn, '=' * 20)
            try:
                args = {p: locals()[p] for p in inspect.signature(fn).parameters}
                fn(**args)
            except pydantic.ValidationError as e:
                print(e)
                rich.print(e.errors())
                print(traceback.format_exc())
                break
    print('PASS')
    ipd.dev.global_timer.report()

def make_tmp_clent_server():
    pppserver, pppbackend, pppclient, ppptestclient = None, None, None, None
    # engine = create_engine(f'sqlite:///{tmpdir}/test.db')
    # pppbackend = ppp.server.Backend(engine, tmpdir)
    # pppserver, pppbackend = ipd.ppp.server.run(port=12345, dburl='postgresql://localhost/ppp')
    # assert not os.path.exists(f'{tmpdir}/test.db')
    pppserver, pppbackend, pppclient = ipd.ppp.server.run(
        port=12346,
        dburl='sqlite:////tmp/test.db',
        # dburl='<memory>',
        workers=1,
        loglevel='warning')
    ppptestclient = TestClient(pppbackend.app)
    ppp.server.defaults.ensure_init_db(pppbackend)
    # pppclient = ipd.ppp.PPPClient('localhost:12345')
    return pppserver, pppbackend, pppclient, ppptestclient

def _test_post_hang():
    # only hangs if pollfileid is a string... any string
    import requests
    url = 'http://localhost:12346/ppp/create/review'
    # body = '''{"id":"40d493e7-fb18-44b5-be1c-a5154ad0c4d8","ispublic":true,"telemetry":false,"ghost":false,"datecreated":"2024-10-13T23:16:57.436648","props":[],"attrs":{},"userid":"24c41db4-7e45-4265-b666-71f477955a01","pollid":"7cf6d8f5-f9d1-4563-99f7-1a95d429dd68","grade":"dislike","comment":"","pollfileid":"2bbfa731-949f-4dfe-8119-94fdae221b8e"}'''
    body = '''{"pollfileid":""}'''
    # ic(url, body)
    response = requests.post(url, body)
    assert response.content.count(b'bad UUID string')

@pytest.mark.fast
def test_pollfiles(pppclient):
    poll = pppclient.newpoll(name='polio', path=ipd.dev.package_testdata_path('ppppdbdir'))
    for f in poll.pollfiles:
        assert f == pppclient.pollfile(pollid=poll.id, fname=f.fname)

@pytest.mark.fast
def test_review(pppclient):
    poll = pppclient.newpoll(name='polio', path=ipd.dev.package_testdata_path('ppppdbdir'))
    assert poll.pollfiles
    poll2 = pppclient.newpoll(name='polio2', path=ipd.dev.package_testdata_path('ppppdbdir'))
    poll3 = pppclient.newpoll(name='polio3', path=ipd.dev.package_testdata_path('ppppdbdir'))
    file = next(iter(poll.pollfiles))
    pppclient.newuser(name='reviewer')
    print([p.name for p in pppclient.users()])
    # print(pppclient.user(name='reviewer'))
    rev = pppclient.newreview(userid='reviewer',
                              pollid=poll.id,
                              pollfileid=file.id,
                              workflowid='Manual',
                              grade='dislike')
    assert pppclient.reviews()[0].user.name == 'reviewer'
    assert os.path.exists(rev.pollfile.permafname)
    # print('\n'.join([f'{f.pollid} {f.fname}' for f in pppclient.pollfiles()]))
    polls = pppclient.polls()
    review = ppp.ReviewSpec(pollid=polls[2].id,
                            pollfileid=polls[2].pollfiles[2].id,
                            grade='superlike',
                            comment='foobar')
    assert review.grade == 'superlike'
    result = pppclient.upload_review(review)
    assert isinstance(result, ipd.ppp.Review)
    # rich.print(pppclient.reviews())
    assert len(pppclient.reviews()) == 2
    result = pppclient.upload(ppp.ReviewSpec(pollid=poll2.id, pollfileid=poll2.pollfiles[2].id, grade='hate'))
    assert isinstance(result, ipd.ppp.Review)
    assert file.fname in [f.fname for f in poll.pollfiles]
    assert pppclient.poll(id=poll.id)
    assert pppclient.pollfile(pollid=poll.id, fname=poll.pollfiles[0].fname)
    print(len(pppclient.pollfile(pollid=poll.id, fname=file.fname).reviews))
    assert 1 == len(pppclient.pollfile(pollid=poll.id, fname=file.fname).reviews)
    assert 3 == len(pppclient.reviews())
    assert 1 == len(list(it.chain(*(f.reviews for f in pppclient.pollfiles(fname=file.fname)))))
    reviews = pppclient.reviews()
    polls = pppclient.polls()
    files = pppclient.pollfiles()

    for i, p in enumerate(pppclient.polls()):
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

    # print([p.name for p in pppclient.polls()])
    # print(len(pppclient.polls(name='foo1')))

@pytest.mark.fast
def test_poll_attr(pppclient):
    poll = pppclient.upload_poll(ppp.PollSpec(name='foo', path=ipd.dev.package_testdata_path('ppppdbdir')))
    # poll.print_full()
    # print(type(poll.id), type(poll.pollfiles[0].pollid))
    # print(poll.id == poll.pollfiles[0].pollid)
    assert all(poll.id == p.pollid for p in poll.pollfiles)
    assert isinstance(poll.pollfiles[0], ppp.PollFile)

# def test_spec_srict_ctor_override():
#     class Foo(pydantic.BaseModel, ipd.crud.StrictFields):
#         bar: int
#
#     a = Foo(bar=6)
#     with pytest.raises(TypeError):
#         b = Foo(baz=6)
#
#     class FooNonstrict(Foo):
#         bar: int
#
#         def __init__(self, zaz, **kw):
#             super().__init__(**kw)
#
#     c = FooNonstrict(zaz=6, bar=6)

@pytest.mark.fast
def test_setattr(pppclient):
    user = pppclient.newuser(name='foo', fullname='bar')
    assert user.fullname == 'bar'
    user.fullname = 'baz'
    usercopy = pppclient.user(id=user.id)
    assert usercopy.fullname == 'baz'
    assert user.fullname == 'baz'

@pytest.mark.fast
def test_pollinfo(pppclient, pppbackend):
    pppbackend.newpoll(name='foo', path='.', ispublic=True, user=pppbackend.newuser(name='foo'))
    pppbackend.newpoll(name='bar', path='.', ispublic=False, user=pppbackend.newuser(name='bar'))
    pppbackend.newpoll(name='baz', path='.', ispublic=True, user=pppbackend.newuser(name='baz'))
    assert len(pppbackend.polls()) == 3
    assert len(pppbackend.pollinfo(user='foo')) == 2
    assert len(pppbackend.pollinfo(user='bar')) == 3
    assert len(pppbackend.pollinfo(user='baz')) == 2
    binfo = pppbackend.pollinfo(user='admin')
    info = pppclient.pollinfo(user='admin')
    assert binfo == info

@pytest.mark.fast
def test_pymolcmdsdict(pppclient):
    pcd = pppclient.pymolcmdsdict()
    assert isinstance(pcd, list)
    assert isinstance(pcd[0], dict)
    cmds = pppclient.pymolcmds()
    cmd = cmds[0]

@pytest.mark.fast
def test_ghost(pppbackend):
    user = pppbackend.newuser(name='jameswoods')
    poll = pppbackend.newpoll(name='foo', path='bar', user=user)
    print(pppbackend.newpollfile)
    file = pppbackend.newpollfile(fname='baz', pollid=poll.id)
    poll.pollfiles.append(file)
    pppbackend.session.commit()
    assert not file.ghost
    print(poll.clear)
    poll.clear(pppbackend)
    assert file.ghost
    assert 1 == len(pppbackend.pollinfo(user=user.name))
    assert not poll.ghost
    pppbackend.remove('poll', poll.id)
    assert poll.ghost
    assert 0 == len(pppbackend.pollinfo(user=user.name))

@pytest.mark.fast
def test_user_backend(pppbackend):
    foo = pppbackend.newuser(name='foo')
    assert pppbackend.iserror(pppbackend.newuser(name='foo'))
    print('!' * 80)
    return

    follower = pppbackend.newuser(name='following1')
    follower.following.append(foo)
    follower.following.append(foo)
    follower.following.append(foo)
    pppbackend.session.commit()
    assert follower in foo.followers
    assert len(foo.followers) == 1
    assert len(follower.following) == 1
    foo.followers = []
    assert follower not in foo.followers
    assert foo not in follower.following

# def test_spec_basics():
#     with pytest.raises(TypeError):
#         ppp.PollSpec(name='foo', path='.', userid='test', ntisearien=1)

@pytest.mark.fast
def test_access_all(pppclient, pppbackend):
    assert 0 == (len(pppclient.polls()))

@pytest.mark.fast
def test_read_root(pppclient):
    assert pppclient.get("/") == {"msg": "Hello World"}

def _test_file_upload(pppclient, pppbackend):
    # pppclient = ppp.PPPClient(ppptestclient)
    path = ipd.dev.package_testdata_path('ppppdbdir')
    spec = ppp.PollSpec(name='usertest1pub', path=path, userid='test', ispublic=True)
    if response := pppclient.upload_poll(spec): print(response)
    localfname = os.path.join(path, '1pgx.cif')
    file = ipd.ppp.PollFileSpec(pollid=spec.id, fname=localfname)
    poll = pppclient.upload(ipd.ppp.PollSpec(name='foo', path='.'))
    assert isinstance(poll, ipd.ppp.Poll), poll
    exists, newfname = pppclient.get('/have/pollfile', fname=file.fname, pollid=pppclient.polls()[-1].id)
    assert file.fname in [f.fname for f in pppclient.pollfiles()]
    assert newfname.endswith('\\ipd\\tests\\data\\ppppdbdir\\1pgx.cif')
    exists, newfname = pppbackend.have_pollfile(fname=localfname, pollid=pppclient.polls()[-1].id)
    # print(exists, f'"{newfname}"')
    assert newfname.endswith('\\ipd\\tests\\data\\ppppdbdir\\1pgx.cif')
    filecontent = Path(localfname).read_text()
    file.filecontent = filecontent
    file.permafname = newfname
    file = ipd.ppp.PollFileSpec(**file.dict())
    # response = pppbackend.create_file(file)
    pppclient.post('/create/pollfilecontents', file)
    print(newfname)
    diff = subprocess.check_output(['diff', localfname, newfname])
    if diff: print(f'diff {localfname} {newfname} {diff}')
    assert not diff
    os.remove(newfname)
    file.filecontent = ''
    files = [file._copy_with_newid() for _ in range(10)]
    pppbackend.create_empty_files(files)
    poll = pppclient.polls(name='usertest1pub')[0]
    print(pppclient.npollfiles())
    # for p in pppclient.pollfiles():
    # print(f'{p.poll.name} {p.fname}')
    assert pppclient.npollfiles() == 15
    assert len(pppclient.pollfiles()) == 15
    pppclient.remove(poll)
    assert len(pppclient.polls()) == 1

@pytest.mark.fast
def test_poll(pppclient, pppbackend):
    path = ipd.dev.package_testdata_path('ppppdbdir')
    pppclient.upload(ppp.UserSpec(name='test1'))

    assert pppclient.user(name='test1').fullname == ''
    pppclient.user(name='test1').fullname = 'fulllname'
    assert pppclient.user(name='test1').fullname == 'fulllname'

    pppclient.upload(ppp.UserSpec(name='test2'))
    pppclient.upload(ppp.UserSpec(name='test3'))
    pppclient.upload_poll(ppp.PollSpec(name='usertest1pub', path=path, userid='test1', ispublic=True))
    pppclient.upload_poll(ppp.PollSpec(name='usertest1pri', path=path, userid='test1', ispublic=False))
    pppclient.upload_poll(ppp.PollSpec(name='usertest2pub', path=path, userid='test2', ispublic=True))
    pppclient.upload_poll(ppp.PollSpec(name='usertest3pri', path=path, userid='test3', ispublic=False))
    assert 3 == len(pppclient.pollinfo(user='test1'))
    assert 2 == len(pppclient.pollinfo(user='test2'))
    assert 3 == len(pppclient.pollinfo(user='test3'))
    assert 2 == len(pppclient.pollinfo(user='sheffler'))
    assert 5 == len(pppclient.pollinfo(user='admin'))

    poll = pppclient.upload_poll(ppp.PollSpec(name='foo1', desc='bar', path=path))
    assert isinstance(poll, ppp.Poll)
    poll = pppclient.upload_poll(ppp.PollSpec(name='foo2', desc='Nntsebar', path=path))
    assert isinstance(poll, ppp.Poll)
    poll = pppclient.upload_poll(
        ppp.PollSpec(name='foo3', desc='barntes', path=path, props=['ligand', 'multichain']))

    assert len(pppclient.polls()) == 7
    polljs = pppclient.get('/polls')
    # print(polljs)
    # polls = [ppp.Poll(**_) for _ in polljs]
    polls = pppbackend.polls()
    # for poll in polls:
    # print(poll, len(poll.files))
    # print(list(poll.files)[:2])

    pfiles = polls[1].pollfiles
    assert len(pfiles) == 3

    poll3 = pppclient.polls()[6]
    # print(poll3)
    assert 'ligand' in poll3.props

    poll = ppp.PollSpec(name='foobar', path=path, props=['ast'])
    # print(poll)
    poll = pppclient.upload_poll(poll)
    assert len(pppclient.polls()) == 8
    poll = pppclient.polls()[6]
    assert isinstance(poll, ipd.ppp.Poll)
    # for p in pppclient.polls():
    # print(p.id, p.name, len(p.files))
    assert len(poll.pollfiles) == 3
    assert isinstance(poll.pollfiles[0], ipd.ppp.PollFile)

    result = pppclient.upload(
        ppp.PymolCMDSpec(name='test', cmdon='show lines', cmdoff='hide lines', userid='test'))
    assert isinstance(result, ppp.PymolCMD)
    result = pppclient.upload(ppp.PymolCMDSpec(name='test2', cmdon='fubar', cmdoff='hide lines',
                                               userid='test')).count('NameError')
    # print(result)
    with pytest.raises(ipd.crud.ClientError):
        result = pppclient.upload(
            ppp.PymolCMDSpec(name='test', cmdon='show lines', cmdoff='hide lines', userid='test'))
    # print(result)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    assert len(pppclient.pymolcmds(user='test')) == 1

if __name__ == '__main__':
    main()

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

rich = ipd.dev.lazyimport('rich', 'Rich', pip=True)
pytest = ipd.dev.lazyimport('pytest', pip=True)

testclient, backend = None, None

def main():
    for fn in [f for n, f in globals().items() if n.startswith('test_')]:
        with tempfile.TemporaryDirectory() as tempdir:
            print('=' * 20, fn, '=' * 20)
            server, backend, client, testclient = make_tmp_clent_server(tempdir)
            try:
                args = {p: locals()[p] for p in inspect.signature(fn).parameters}
                print(args)
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
    server, backend, client = ipd.ppp.server.run(port=12345, dburl=f'sqlite:///{tempdir}/test.db')
    testclient = TestClient(backend.app)
    # ppp.server.defaults.ensure_init_db(backend)
    return server, backend, client, testclient

def _test_ghost(backend):
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
    assert 1 == len(backend.pollinfo(user=user.name))

def _test_user_backend(backend):
    foo = backend.newuser(name='foo')
    assert not backend.newuser(name='foo')
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

def _test_spec_basics():
    with pytest.raises(TypeError):
        ppp.PollSpec(name='foo', path='.', userid='test', ntisearien=1)

def _test_access_all(client, backend):
    print(len(client.polls()))

def _test_read_root(client):
    assert client.get("/") == {"msg": "Hello World"}

def _test_file_upload(client, backend):
    # client = ppp.PPPClient(testclient)
    path = ipd.testpath('ppppdbdir')
    spec = ppp.PollSpec(name='usertest1pub', path=path, userid='test', ispublic=True)
    if response := client.upload_poll(spec): print(response)
    localfname = os.path.join(path, '1pgx.cif')
    file = ipd.ppp.PollFileSpec(pollid=2, fname=localfname)
    exists, newfname = client.get('/have/pollfile', fname=file.fname, pollid=client.npolls())
    assert file.fname in [f.fname for f in client.pollfiles()]
    assert newfname.endswith('\\ipd\\tests\\data\\ppppdbdir\\1pgx.cif')
    exists, newfname = backend.have_pollfile(fname=localfname, pollid=client.npolls())
    assert exists == (file.fname in [f.fname for f in client.pollfiles()])
    assert newfname.endswith('\\ipd\\tests\\data\\ppppdbdir\\1pgx.cif')
    filecontent = Path(localfname).read_text()
    file.filecontent = filecontent
    file.permafname = newfname
    file = ipd.ppp.PollFileSpec(**file.dict())
    # response = backend.create_file(file)
    client.post('/create/pollfile', file)
    diff = subprocess.check_output(['diff', localfname, newfname])
    if diff: print(f'diff {localfname} {newfname} {diff}')
    assert not diff
    file.filecontent = ''
    files = [file for _ in range(10)]
    backend.create_empty_files(files)
    poll = client.polls(name='usertest1pub')[0]
    # print(client.npollfiles())
    # for p in client.pollfiles():
    # print(f'{p.poll.name} {p.fname}')
    assert client.npollfiles() == 16
    assert len(client.pollfiles()) == 16
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

    assert len(client.polls()) == 8
    assert client.get('/poll?id=2')['id'] == 2
    polljs = client.get('/polls')
    # print(polljs)
    # polls = [ppp.Poll(**_) for _ in polljs]
    polls = backend.polls()
    assert len(polls) == 8
    # for poll in polls:
    # print(poll, len(poll.files))
    # print(list(poll.files)[:2])

    pfiles = polls[1].pollfiles
    assert len(pfiles) == 4

    poll3 = client.poll(id=8)
    # print(poll3)
    assert 'ligand' in poll3.props

    poll = ppp.PollSpec(name='foobar', path=path, props=['ast'])
    # print(poll)
    poll = client.upload_poll(poll)
    assert len(client.polls()) == 9
    poll = client.poll(id=8)
    assert isinstance(poll, ipd.ppp.Poll)
    # for p in client.polls():
    # print(p.id, p.name, len(p.files))
    assert len(poll.pollfiles) == 4
    assert isinstance(poll.pollfiles[0], ipd.ppp.PollFile)
    for i in range(1, 4):
        assert backend.poll(dict(id=i)).pollfiles[0].poll.id == i

    fname = ipd.testpath('ppppdbdir/1pgx.cif')
    client.newuser(name='reviewer')
    # print([p.name for p in client.users()])
    # print(client.user(name='reviewer'))
    rev = client.newreview(userid='reviewer', pollid=poll.id, pollfileid=fname, workflowid=1, grade='dislike')
    assert client.reviews()[0].user.name == 'reviewer'
    assert os.path.exists(rev.pollfile.permafname)
    # print('\n'.join([f'{f.pollid} {f.fname}' for f in client.pollfiles()]))
    review = ppp.ReviewSpec(pollid=2, pollfileid=fname, grade='superlike', comment='foobar')
    assert review.grade == 'superlike'
    result = client.upload_review(review)
    assert isinstance(result, ipd.ppp.Review)
    # rich.print(client.reviews())
    assert len(client.reviews()) == 2
    result = client.upload(ppp.ReviewSpec(pollid=3, pollfileid=fname, grade='hate'))
    assert isinstance(result, ipd.ppp.Review)

    assert 1 == len(client.pollfile(id=poll.id, fname=fname).reviews)
    assert 3 == len(list(it.chain(*(f.reviews for f in client.pollfiles(fname=fname)))))
    reviews = client.reviews()
    polls = client.polls()
    files = client.pollfiles()

    assert reviews[0].poll.id == 8
    assert reviews[0].pollfile.id == 28

    assert isinstance(files[0], ppp.PollFile)
    assert isinstance(polls[2].pollfiles[0], ppp.PollFile)
    assert isinstance(reviews[2].pollfile, ppp.PollFile)
    assert isinstance(reviews[0], ppp.Review)
    assert isinstance(polls[2].reviews[0], ppp.Review)
    assert isinstance(files[1].poll, ppp.Poll)

    for r in reviews:
        print('permafname', r.pollfile.permafname)
        print(r.pollfile)
        assert os.path.exists(r.pollfile.permafname)

    print([p.name for p in client.polls()])
    print(len(client.polls(name='foo1')))

    assert len(client.polls(name='foo1')) == 1

    result = client.upload(
        ppp.PymolCMDSpec(name='test', cmdon='show lines', cmdoff='hide lines', userid='test'))
    assert isinstance(result, ppp.PymolCMD)
    result = client.upload(ppp.PymolCMDSpec(name='test2', cmdon='fubar', cmdoff='hide lines',
                                            userid='test')).count('NameError')
    print(result)
    result = client.upload(
        ppp.PymolCMDSpec(name='test', cmdon='show lines', cmdoff='hide lines', userid='test'))
    print(result)

    assert len(client.pymolcmds(user='test')) == 1

if __name__ == '__main__':
    main()

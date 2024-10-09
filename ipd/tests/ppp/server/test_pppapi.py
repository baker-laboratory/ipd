import sys
import ipd
from ipd import ppp
import tempfile
from pathlib import Path
import os
import subprocess
from fastapi.testclient import TestClient
from sqlmodel import create_engine
from icecream import ic

rich = ipd.dev.lazyimport('rich', 'Rich', pip=True)
pytest = ipd.dev.lazyimport('pytest', pip=True)

testclient, backend = None, None

def main():
    test_spec_basics()
    for fn in [test_read_root, test_file_upload, test_poll, test_access_all]:
        with tempfile.TemporaryDirectory() as td:
            print('=' * 20, fn, '=' * 20)
            server, backend, testclient = make_tmp_clent_server(td)
            fn(testclient, backend)
            server.stop()
    print('PASS')
    ipd.dev.global_timer.report()

def test_spec_basics():
    with pytest.raises(TypeError):
        ppp.PollSpec(name='foo', path='.', userid='test', ntisearien=1)

def test_access_all(testclient, backend):
    client = testclient
    # client = ppp.PPPClient(testclient)
    print(len(client.polls()))

def make_tmp_clent_server(tempdir):
    # engine = create_engine(f'sqlite:///{tempdir}/test.db')
    # backend = ppp.server.Backend(engine, tempdir)
    # testclient = TestClient(backend.app)
    # server, backend = ipd.ppp.server.run(port=12345, dburl='postgresql://localhost/ppp')
    assert not os.path.exists(f'{tempdir}/test.db')
    server, backend, testclient = ipd.ppp.server.run(port=12345, dburl=f'sqlite:///{tempdir}/test.db')
    ppp.server.defaults.ensure_init_db(backend)
    return server, backend, testclient

def test_read_root(testclient, backend):
    assert testclient.get("/") == {"msg": "Hello World"}

def test_file_upload(testclient, backend):
    client = testclient
    # client = ppp.PPPClient(testclient)
    path = ipd.testpath('ppppdbdir')
    spec = ppp.PollSpec(name='usertest1pub', path=path, userid='test', ispublic=True)
    if response := client.upload_poll(spec): print(response)
    localfname = os.path.join(path, '1pgx.cif')
    file = ipd.ppp.PollFileSpec(pollid=2, fname=localfname)
    exists, newfname = client.get('/have/pollfile', fname=file.fname, pollid=client.npolls())
    print('new', file.fname)
    for f in client.pollfiles():
        print(f.fname)
    assert exists == (file.fname in [f.fname for f in client.pollfiles()])
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

def test_poll(testclient, backend):
    client = testclient
    # client = ppp.PPPClient(testclient)
    path = ipd.testpath('ppppdbdir')
    client.upload(ppp.UserSpec(name='test1'))
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

    response = client.upload_poll(ppp.PollSpec(name='foo1', desc='bar', path=path))
    assert not response, response
    response = client.upload_poll(ppp.PollSpec(name='foo2', desc='Nntsebar', path=path))
    assert not response, response
    response = client.upload_poll(
        ppp.PollSpec(name='foo3', desc='barntes', path=path, props=['ligand', 'multichain']))
    assert not response, response

    assert len(client.polls()) == 8
    assert testclient.get('/poll?id=2')['id'] == 2
    polljs = testclient.get('/polls')
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
    result = client.upload_poll(poll)
    assert not result, result
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
    client.upload_review(
        ppp.ReviewSpec(fname=fname,
                       userid='test',
                       pollid=poll.id,
                       fileid=poll.pollfiles[0].id,
                       grade='dislike'))
    assert client.reviews()[0].user.name == 'test'
    review = ppp.ReviewSpec(pollid=1, fname=fname, grade='superlike')
    assert review.grade == 'superlike'
    client.upload_review(review)
    assert len(client.reviews()) == 2
    client.upload_review(ppp.ReviewSpec(pollid=3, fname=fname, grade='f'))

    assert len(client.reviews_for_fname(fname)) == 3

    reviews = client.reviews()
    polls = client.polls()
    files = client.files()
    cmds = client.pymolcmds()

    assert reviews[0].poll.id == 1
    assert reviews[0].file.id == 2
    print(len(polls))
    for p in polls:
        print(p)
    assert len(polls[3].files) == 3

    assert isinstance(files[0], ppp.File)
    assert isinstance(polls[2].files[0], ppp.File)
    assert isinstance(reviews[2].file, ppp.File)
    assert isinstance(reviews[0], ppp.Review)
    assert isinstance(polls[2].reviews[0], ppp.Review)
    assert isinstance(files[1].reviews[0], ppp.Review)
    assert isinstance(files[1].poll, ppp.Poll)

    assert not client.upload(ppp.PymolCMDSpec(name='test', cmdon='show lines', cmdoff='hide lines'))
    assert client.upload(ppp.PymolCMDSpec(name='test2', cmdon='fubar', cmdoff='hide lines')).count('NameError')
    assert client.upload(ppp.PymolCMDSpec(name='test', cmdon='show lines',
                                          cmdoff='hide lines')).count('duplicate')
    assert len(client.pymolcmds()) == 1

    for r in reviews:
        assert os.path.exists(r.permafname)

    print([p.name for p in client.polls()])
    print(len(client.polls(name='foo1')))

    assert len(client.polls(name='foo1')) == 1

if __name__ == '__main__':
    main()

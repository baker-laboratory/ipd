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

testclient, pppserver = None, None

def main():
    for fn in [test_read_root, test_file_upload, test_poll, test_access_all]:
        with tempfile.TemporaryDirectory() as td:
            testclient, pppserver = make_tmp_clent_server(td)
            fn(testclient, pppserver)
    print('PASS')
    ipd.dev.global_timer.report()

@ipd.dev.timed
def test_access_all(testclient, pppserver):
    client = ppp.PPPClient(testclient)
    print(len(client.polls()))

@ipd.dev.timed
def make_tmp_clent_server(tempdir):
    engine = create_engine(f'sqlite:///{tempdir}/test.db')
    pppserver = ppp.server.Backend(engine, tempdir)
    testclient = TestClient(pppserver.app)
    return testclient, pppserver

def test_read_root(testclient, pppserver):
    response = testclient.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

def test_file_upload(testclient, pppserver):
    client = ppp.PPPClient(testclient)
    path = ipd.testpath('ppppdbdir')
    if response := client.upload_poll(ppp.PollSpec(name='usertest1pub', path=path, user='user1',
                                                   ispublic=True)):
        print(response)
    localfname = os.path.join(path, '1pgx.cif')
    file = ipd.ppp.FileSpec(pollid=1, fname=localfname)
    exists, newfname = client.post('/have/file', file)
    assert not exists
    assert newfname.endswith('\\ipd\\tests\\data\\ppppdbdir\\1pgx.cif')
    exists, newfname = pppserver.have_file(file)
    assert not exists
    assert newfname.endswith('\\ipd\\tests\\data\\ppppdbdir\\1pgx.cif')
    filecontent = Path(file.fname).read_text()
    file.filecontent = filecontent
    file.permafname = newfname
    file = ipd.ppp.FileSpec(**file.dict())
    # response = pppserver.create_file(file)
    client.post('/create/file', file)
    diff = subprocess.check_output(['diff', localfname, newfname])
    if diff: print(f'diff {localfname} {newfname} {diff}')
    assert not diff
    file.filecontent = ''
    files = [file for _ in range(10)]
    pppserver.create_empty_files(files)
    poll = client.polls(name='usertest1pub')[0]
    assert len(poll.files) == 13
    assert len(client.files()) == 13
    client.remove(poll)
    assert len(client.polls()) == 0

@ipd.dev.timed
def test_poll(testclient, pppserver):
    client = ppp.PPPClient(testclient)
    path = ipd.testpath('ppppdbdir')
    client.upload_poll(ppp.PollSpec(name='usertest1pub', path=path, user='user1', ispublic=True))
    client.upload_poll(ppp.PollSpec(name='usertest1pri', path=path, user='user1', ispublic=False))
    client.upload_poll(ppp.PollSpec(name='usertest2pub', path=path, user='user2', ispublic=True))
    client.upload_poll(ppp.PollSpec(name='usertest3pri', path=path, user='user3', ispublic=False))
    assert 3 == len(client.pollinfo(user='user1'))
    assert 2 == len(client.pollinfo(user='user2'))
    assert 3 == len(client.pollinfo(user='user3'))

    response = client.upload_poll(ppp.PollSpec(name='foo1', desc='bar', path=path))
    assert not response, response
    response = client.upload_poll(ppp.PollSpec(name='foo2', desc='Nntsebar', path=path))
    assert not response, response
    response = client.upload_poll(
        ppp.PollSpec(name='foo3', desc='barntes', path=path, props=['ligand', 'multichain']))
    assert not response, response

    assert len(client.polls()) == 7
    assert testclient.get('/poll2').json()['id'] == 2
    polljs = testclient.get('/polls').json()
    # print(polljs)
    # polls = [ppp.Poll(**_) for _ in polljs]
    polls = pppserver.polls()
    assert len(polls) == 7
    # for poll in polls:
    # print(poll, len(poll.files))
    # print(list(poll.files)[:2])

    pfiles = polls[1].files
    assert len(pfiles) == 3
    for f in pfiles:
        f2 = testclient.get('/poll1/fname?trackseen=True').json()
        assert f2['fname'] == f.fname
        assert len(f2['next']) <= 10
    assert testclient.get('poll1/fname?trackseen=True').json()['fname'] is None

    totne = 0
    for f in polls[2].files:
        f2 = testclient.get('/poll2/fname?shuffle=True&trackseen=True').json()
        # ic(f2)
        totne += f2['fname'] != f
        assert len(f2['next']) <= 10
    # assert totne > 50
    assert testclient.get('poll2/fname?shuffle=True&trackseen=True').json()['fname'] is None

    poll3 = client.poll(7)
    # print(poll3)
    assert 'ligand' in poll3.props

    poll = ppp.PollSpec(name='foobar', path=path, props=['ast'])
    # print(poll)
    result = client.upload_poll(poll)
    assert not result, result
    assert len(client.polls()) == 8
    poll = client.poll(8)
    # for p in client.polls():
    # print(p.id, p.name, len(p.files))
    assert len(poll.files) == 3
    for i in range(1, 4):
        assert pppserver.poll(i).files[0].poll.id == i

    fname = ipd.testpath('ppppdbdir/1pgx.cif')
    client.upload_review(ppp.ReviewSpec(fname=fname, user='bar', pollid=1, grade='C'))
    assert testclient.get('/reviews').json()[0]['user'] == 'bar'
    review = ppp.ReviewSpec(pollid=1, fname=fname, grade='A')
    assert review.grade == 'A'
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

import ipd
import tempfile
from fastapi.testclient import TestClient
from sqlmodel import create_engine
from icecream import ic

rich = ipd.dev.pipimport('rich', 'Rich')
from rich import print

testclient, pppserver = None, None

def main():
    with tempfile.TemporaryDirectory() as td:
        testclient, pppserver = make_tmp_clent_server(td)
        test_read_root(testclient, pppserver)
        test_poll(testclient, pppserver)
        print('PASS')

def make_tmp_clent_server(tempdir):
    engine = create_engine(f'sqlite:///{tempdir}/test.db')
    pppserver = ipd.ppp.server.Server(engine, tempdir)
    testclient = TestClient(pppserver.app)
    return testclient, pppserver

def test_read_root(testclient, pppserver):
    response = testclient.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

def test_poll(testclient, pppserver):
    client = ipd.ppp.PPPClient(testclient)

    assert testclient.post('/poll').status_code == 422
    path = ipd.testpath('ppppdbdir')
    assert testclient.post('/poll', json=dict(name='foo1', desc='bar', path=path))
    assert testclient.post('/poll', json=dict(name='foo2', desc='Nntsebar', path=path))
    assert testclient.post('/poll',
                           json=dict(name='foo3', desc='barntes', path=path, props=['ligand', 'multichain']))

    assert testclient.get('/poll2').json()['pollid'] == 2
    polljs = testclient.get('/polls').json()
    # print(polljs)
    # polls = [ipd.ppp.Poll(**_) for _ in polljs]
    polls = pppserver.polls()
    assert len(polls) == 3
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

    poll3 = client.poll(3)
    # print(poll3)
    assert 'ligand' in poll3.props

    poll = ipd.ppp.PollUpload(path=path, props=['ast'])
    # print(poll)
    client.post(poll)
    poll = client.poll(3)
    assert len(poll.fids) == 3
    for i in range(1, 4):
        assert pppserver.poll(i).files[0].poll.pollid == i

    fname = ipd.testpath('ppppdbdir/fake1.pdb')
    testclient.post('/poll1/review', json=dict(fname=fname, user='bar', grade='C', data={}))
    assert testclient.get('/reviews').json()[0]['user'] == 'bar'
    review = ipd.ppp.ReviewUpload(pollid=1, fname=fname, grade='A')
    assert review.grade == 'A'
    client.post_review(review)
    assert len(client.reviews()) == 2
    client.post_review(ipd.ppp.ReviewUpload(pollid=3, fname=fname, grade='f'))

    assert len(client.reviews_for_fname(fname)) == 3
    assert len(client.reviews_for_pollid(1)) == 2
    assert len(client.reviews_for_fileid(1)) == 2



def test_pppapi():

    assert 0

if __name__ == '__main__':
    main()

import tempfile
from ipd.ppp.server import PPPServer, Poll
from fastapi.testclient import TestClient
from sqlmodel import create_engine
from icecream import ic

client, server = None, None

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

def test_poll():
    assert client.post('/poll').status_code == 422
    path = '/home/sheffler/project/rfdsym/hilvert/pymol_saves'
    assert client.post('/poll', json=dict(name='foo', desc='bar', path=path))
    assert client.post('/poll', json=dict(name='foo', desc='bar', path=path))
    assert client.post('/poll', json=dict(name='foo', desc='bar', path=path))

    assert client.get('/poll2').json()[0]['pollid'] == 2
    polljs =client.get('/poll').json()
    print(polljs)
    polls = [Poll(**_) for _ in polljs]
    assert len(polls) == 3
    for poll in polls:
        print(poll, len(poll.files))
        # print(list(poll.files)[:2])

    for f in polls[1].files:
        f2 = client.get('/poll1/file?trackseen=True').json()
        assert f2['file'] == f
        assert len(f2['next']) <= 10
    assert client.get('poll1/file?trackseen=True').json()['file'] is None

    totne = 0
    for f in polls[2].files:
        f2 = client.get('/poll2/file?shuffle=True&trackseen=True').json()
        # ic(f2)
        totne += f2['file'] != f
        assert len(f2['next']) <= 10
    assert totne > 50
    assert client.get('poll2/file?shuffle=True&trackseen=True').json()['file'] is None

    client.post('/poll1/review',json=dict(file='foo', user='bar', grade='C', data={}))

def main():
    global client, server
    with tempfile.TemporaryDirectory() as td:
        engine = create_engine(f'sqlite:///{td}/test.db')
        server = PPPServer(engine, td)
        client = TestClient(server.app)
        test_read_root()
        test_poll()
        print('PASS')

def test_pppapi():

    assert 0

if __name__ == '__main__':
    main()

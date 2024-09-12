from ipd.ppp.server import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}

def main():
    test_read_main()
    print('PASS')

def test_pppapi():

    assert 0

if __name__ == '__main__':
    main()

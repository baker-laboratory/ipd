import os
import sys
import uvicorn
from ipd.ppp.server import PPPServer
from sqlmodel import create_engine

def main():
    port = int(sys.argv[1])
    datadir = os.path.realpath(sys.argv[2])
    engine = create_engine(f'sqlite:///{datadir}/ppp.db')
    server = PPPServer(engine, datadir)
    server.app.mount("/ppp", server.app)  # your app routes will now be /app/{your-route-here}
    uvicorn.run(server.app, host="0.0.0.0", port=port, log_level="info")

if __name__ == '__main__':
    main()

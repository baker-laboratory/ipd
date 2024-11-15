import contextlib
import functools
import os
import signal
import threading
import time

import sqlmodel
import uvicorn

import ipd

class Server(uvicorn.Server):
    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self, *_):
        # print('shutting down server')
        self.should_exit = True
        self.thread.join()

        # sys.exit()
class CrudRun:
    def __getitem__(self, BackFront=tuple[ipd.crud.BackendBase, ipd.crud.ClientBase]):
        Backend, Client = BackFront
        return functools.partial(self.runserver, Client=Client, Backend=Backend)

    def runserver(
        self,
        port,
        dburl=None,
        datadir='./server_datadir',
        loglevel='warning',
        local=False,
        workers=1,
        background=True,
        Backend: ipd.crud.BackendBase = None,  # type: ignore
        Client: ipd.crud.ClientBase = None,  # type: ignore
        **kw,
    ):
        datadir = os.path.abspath(os.path.expanduser(datadir))
        dburl = dburl or f'sqlite:///{datadir}/{self.Backend.mountpoint}.db'  # type: ignore
        if not dburl.count('://'): dburl = f'sqlite:///{dburl}'
        os.makedirs(datadir, exist_ok=True)
        # print(f'creating db engine from url: \'{dburl}\'')
        engine = sqlmodel.create_engine(dburl)
        backend = Backend(engine, datadir)
        backend.app.mount(f'/{Backend.mountpoint}', backend.app)
        # if not local: pymol_launch()
        config = uvicorn.Config(
            backend.app,
            host='127.0.0.1' if local else '0.0.0.0',
            port=port,
            log_level=loglevel,
            reload=False,
            loop='uvloop',
            workers=workers,
        )
        if background:
            server = Server(config=config)
            server.run_in_thread()
            with contextlib.suppress(ValueError):
                signal.signal(signal.SIGINT, server.stop)
            for _ in range(5000):
                if server.started: break
                time.sleep(0.001)
            else:
                raise RuntimeError('server failed to start')
            client = Client(f'127.0.0.1:{port}')
            backend.add_defaults(**kw)
            # print('server', socket.gethostname())
            return server, backend, client
        else:
            server = uvicorn.Server(config)
            server.run()

run = CrudRun()

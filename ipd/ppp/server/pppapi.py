import sys
import os
from datetime import datetime
import contextlib
import threading
import time
from uuid import UUID
import traceback
import socket
import operator
import ipd
from ipd import ppp
import signal
import rich

fastapi = ipd.lazyimport('fastapi', 'fastapi[standard]', pip=True)
pydantic = ipd.lazyimport('pydantic', pip=True)
sqlmodel = ipd.lazyimport('sqlmodel', pip=True)
sqlalchemy = ipd.lazyimport('sqlalchemy', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
pymol = ipd.lazyimport('pymol', 'pymol-bundle', mamba=True, channels='-c schrodinger')
uvicorn = ipd.dev.lazyimport('uvicorn', 'uvicorn[standard]', pip=True)
# profile = ipd.dev.timed
profile = lambda f: f

_SERVERMODE = 'ppp' == socket.gethostname()

def servermode():
    return _SERVERMODE

def set_servermode(isserver):
    global _SERVERMODE
    _SERVERMODE = isserver
    # print('_SERVERMODE MODE')

@profile
class PPPBackend(ipd.crud.backend.BackendBase, models=ipd.ppp.spec_models):
    def __init__(self, engine, datadir):
        super().__init__(engine)
        self.datadir = datadir
        set_servermode(True)

    def init_routes(self):
        self.route('/api', self.root, methods=['GET'])
        self.route('/api/create/pollfilecontents', self.create_file_with_content, methods=['POST'])
        self.route('/api/create/pollfiles', self.create_empty_files, methods=['POST'])
        self.route('/api/have/pollfile', self.have_pollfile, methods=['GET'])
        self.route('/api/pollinfo', self.pollinfo, methods=['GET'])

    def initdb(self):
        super().initdb()
        ppp.server.defaults.ensure_init_db(self)

    def root(self) -> None:
        return dict(msg='Hello World')

    def useridmap(self):
        query = 'SELECT id,name FROM dbuser WHERE NOT dbuser.ghost'
        idname = self.session.execute(sqlalchemy.text(query)).fetchall()
        return dict(idname), {name: id for id, name in idname}

    def pollinfo(self, user=None):
        query = ('SELECT TB.id,TB.name AS name,dbuser.name AS user,TB.desc,TB.sym,TB.ligand,'
                 'TB.nchain FROM TB JOIN dbuser ON TB.userid = dbuser.id')
        return self._get_table_info(query, user, 'dbpoll')

    def cmdinfo(self, user=None):
        query = ('SELECT TB.id,TB.name AS name,dbuser.name AS user,TB.desc,TB.sym,TB.ligand,'
                 'TB.minchains,TB.maxchains FROM TB JOIN dbuser ON TB.userid = dbuser.id')
        return self._get_table_info(query, user, 'dbpymolcmd')

    def _get_table_info(self, query, user, table):
        if user == 'None': user = None
        if user and user != 'admin':
            query += f' WHERE NOT TB.ghost AND (TB.ispublic OR dbuser.name = \'{user}\')'
        query = query.replace('TB', table)
        # print(query)
        result = self.session.execute(sqlalchemy.text(f'{query};')).fetchall()
        result = list(map(list, result))
        return result

    def reviews_fname(self, fname):
        fname = fname.replace('__DIRSEP__', '/')
        files = self.session.exec(sqlmodel.select(DBPollFile).where(DBPollFile.fname == fname))
        rev = ordset.OrderedSet()
        for f in files:
            rev |= f.reviews
        return list(rev)

    def have_pollfile(self, pollid: UUID, fname: str):
        poll = self.poll(dict(id=pollid))
        assert poll, f'invalid pollid {pollid} {self.npolls()}'
        newfname = self.permafname_name(poll, fname)
        return os.path.exists(newfname), newfname

    def permafname_name(self, poll, fname):
        pollname = poll.name.replace(' ', '_').replace('/', '\\')
        path = os.path.join(self.datadir, 'poll', f'{pollname}__{poll.id}', 'reviewed')
        os.makedirs(path, exist_ok=True)
        newfname = os.path.join(path, fname.replace('/', '\\'))
        return newfname

    def create_file_with_content(self, file: 'DBPollFile'):
        assert file.filecontent
        assert file.permafname
        mode = 'wb' if file.permafname.endswith('.bcif') else 'w'
        with open(file.permafname, mode) as out:
            out.write(file.filecontent)

    def create_empty_files(self, files: list[ppp.PollFileSpec]):
        # print('CREATE empty files', len(files))
        for file in files:
            assert not file.filecontent.strip()
            self.session.add(DBPollFile(**file.dict()))
        self.session.commit()

class Server(uvicorn.Server):
    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self, *_):
        # print('shutting down server')
        self.should_exit = True
        self.thread.join()
        # sys.exit()

def pymol_launch():
    stdout = sys.stdout
    stderr = sys.stderr
    pymol.finish_launching(['pymol', '-xiqckK'])
    sys.stdout = stdout
    sys.stderr = stderr
    pymol.cmd.set('suspend_updates', 'on')

@profile
def run(
    port,
    dburl=None,
    datadir='~/.config/ppp/localserver/data',
    loglevel='warning',
    local=False,
    workers=1,
    background=True,
    **kw,
):
    datadir = os.path.abspath(os.path.expanduser(datadir))
    dburl = dburl or f'sqlite:///{datadir}/ppp.db'
    if not dburl.count('://'): dburl = f'sqlite:///{dburl}'
    os.makedirs(datadir, exist_ok=True)
    # print(f'creating db engine from url: \'{dburl}\'')
    engine = sqlmodel.create_engine(dburl)
    backend = PPPBackend(engine, datadir)
    backend.app.mount("/ppp", backend.app)
    # if not local: pymol_launch()
    config = uvicorn.Config(
        backend.app,
        host='127.0.0.1' if local else '0.0.0.0',
        port=port,
        log_level=loglevel,
        reload=False,
        # reload_dirs=[
        # os.path.join(ipd.proj_dir, 'ipd/ppp'),
        # os.path.join(ipd.proj_dir, 'ipd/ppp/server'),
        # ],
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
        client = ppp.PPPClient(f'127.0.0.1:{port}')
        assert ipd.ppp.get_hack_fixme_global_client()
        ppp.server.defaults.add_defaults(**kw)
        # print('server', socket.gethostname())
        return server, backend, client
    else:
        server = uvicorn.Server(config)
        import pdb
        pdb.set_trace()
        server.run()

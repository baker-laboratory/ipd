import sys
import os
import asyncio
import random
from datetime import datetime
import contextlib
import threading
import time
import traceback
import functools
import socket
import operator
from typing import Optional, Annotated
import ipd
from ipd import ppp
import signal
from rich import print
from ipd.ppp.server.dbmodels import *

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

python_type_to_sqlalchemy_type = {
    str: sqlalchemy.String,
    int: sqlalchemy.Integer,
    float: sqlalchemy.Float,
    bool: sqlalchemy.Boolean,
    # datetime.date: sqlalchemy.Date,
    datetime: sqlalchemy.DateTime,
    # datetime.time: sqlalchemy.Time,
    dict: sqlalchemy.JSON,
    list: sqlalchemy.ARRAY,
    # decimal.Decimal: sqlalchemy.Numeric
}

@profile
class Backend:
    def __init__(self, engine, datadir):
        self.engine = engine
        self.datadir = datadir
        self.session = sqlmodel.Session(self.engine)
        sqlmodel.SQLModel.metadata.create_all(self.engine)
        self.router = fastapi.APIRouter()
        route = self.router.add_api_route
        for model in backend_model:
            route(f'/{model}', getattr(self, model), methods=['GET'])
            route(f'/{model}s', getattr(self, f'{model}s'), methods=['GET'])
            route(f'/n{model}s', getattr(self, f'n{model}s'), methods=['GET'])
            route(f'/create/{model}', getattr(self, f'create_{model}'), methods=['POST'])

        route('/', self.root, methods=['GET'])
        route('/create/pollfilecontents', self.create_file_with_content, methods=['POST'])
        route('/create/pollfiles', self.create_empty_files, methods=['POST'])
        route('/getattr/{thing}/{id}/{attr}', self.getattr, methods=['GET'])
        route('/setattr/{thing}/{id}/{attr}', self.setattr, methods=['POST'])
        route('/have/pollfile', self.have_pollfile, methods=['GET'])
        route('/pollinfo', self.pollinfo, methods=['GET'])
        # route('/poll{id}', self.poll, methods=['GET'])
        # route('/poll{id}/fids', self.poll_fids, methods=['GET'])
        # route('/poll{id}/fname', self.poll_file, methods=['GET'])
        route('/remove/{thing}/{id}', self.remove, methods=['GET'])
        route('/gitstatus/{header}/{footer}', ipd.dev.git_status, methods=['GET'])
        self.app = fastapi.FastAPI()
        self.app.include_router(self.router)
        # ipd.dev.git_status(header='server git status', footer='end', printit=True)
        ppp.server.defaults.ensure_init_db(self)

        @self.app.exception_handler(Exception)
        def validation_exception_handler(request: fastapi.Request, exc: Exception):
            exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
            print('!' * 80)
            print(exc_str)
            print('!' * 80)
            result = f'{"!"*80}\n{exc_str}\nSTACK:\n{traceback.format_exc()}\n{"!"*80}'
            return fastapi.responses.JSONResponse(content=exc_str,
                                                  status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY)

    def remove(self, thing, id):
        thing = self.select(thing, id=id, _single=True)
        thing.ghost = True

    def actually_remove(self, thing, id):
        thing = self.select(thing, id=id, _single=True)
        thing.clear(self)
        self.session.delete(thing)
        self.session.commit()

    def root(self) -> None:
        return dict(msg='Hello World')

    def getattr(self, thing, id: int, attr):
        cls = backend_model[thing]
        thing = self.session.exec(sqlmodel.select(cls).where(cls.id == id)).one()
        if not thing: raise ValueErrors(f'no {cls} id {id} found in database')
        thingattr = getattr(thing, attr)
        if thingattr is None: raise AttributeError(f'db {cls} attr {attr} is None in instance {repr(thing)}')
        return thingattr

    async def setattr(self, request: fastapi.Request, thing: str, id: int, attr: str):
        cls = backend_model[thing]
        thing = self.session.exec(sqlmodel.select(cls).where(cls.id == id)).one()
        if not thing: raise ValueErrors(f'no {cls} id {id} found in database')
        assert thing.model_fields[attr].annotation in (int, float, str)
        body = (await request.body()).decode()
        # print(type(thing), body)
        setattr(thing, attr, body)
        self.session.add(thing)
        self.session.commit()

    def select(self, cls, _count: bool = False, _single=False, user=None, **kw):
        # print('select', cls, kw)
        if isinstance(cls, str): cls = backend_model[cls]
        selection = sqlalchemy.func.count(cls.id) if _count else cls
        statement = sqlmodel.select(selection)
        for k, v in kw.items():
            op = operator.eq
            if k.endswith('not'): k, op = k[:-3], operator.ne
            if v is not None:
                # print('select where', cls, k, v)
                statement = statement.where(op(getattr(cls, k), v))
        if user: statement = statement.where(getattr(cls, 'userid') == self.user(dict(name=user)).id)
        statement = statement.where(getattr(cls, 'ghost') == False)
        if _count: return int(self.session.exec(statement).one())
        thing = list(self.session.exec(statement))
        if _single:
            assert len(thing) == 1
            thing = thing[0]
        return thing

    def fix_date(self, x):
        if hasattr(x, 'datecreated') and isinstance(x.datecreated, str):
            x.datecreated = datetime.strptime(x.datecreated, ppp.DATETIME_FORMAT)
        if hasattr(x, 'enddate') and isinstance(x.enddate, str):
            x.enddate = datetime.strptime(x.enddate, ppp.DATETIME_FORMAT)

    def validate_and_add_to_db(self, thing, replace=False) -> str:
        # print('validate_and_add_to_db', replace)
        self.fix_date(thing)
        self.session.add(thing)
        self.session.commit()
        result = ''
        try:
            try:
                thing = thing.validated_with_backend(self)
            except AssertionError as e:
                self.ssesion.delete(thing)
                return f'error in validate_and_add_to_db: {e}'
        except DuplicateError as e:
            if replace:
                for oldthing in e.conflict:
                    print('server deleted', type(oldthing), oldthing.name3)
                    oldthing.clear(self)
                    self.session.delete(oldthing)
            else:
                thing.clear(self)
                self.session.delete(thing)
                result = 'duplicate'
        self.session.commit()
        return result

    def useridmap(self):
        query = 'SELECT id,name FROM dbuser WHERE NOT dbuser.ghost'
        idname = self.session.execute(sqlalchemy.text(query)).fetchall()
        return dict(idname), {name: id for id, name in idname}

    def pollinfo(self, user=None):
        # print(f'server pollinfo {user}')
        query = ('SELECT dbpoll.id,dbpoll.name,dbuser.name,dbpoll.desc,dbpoll.sym,dbpoll.ligand,'
                 'dbpoll.nchain FROM dbpoll JOIN dbuser ON dbpoll.userid==dbuser.id')
        if user and user != 'admin':
            query += f' WHERE NOT dbpoll.ghost AND (dbpoll.ispublic OR dbuser.name="{user}")'
        result = self.session.execute(sqlalchemy.text(f'{query};')).fetchall()
        result = list(map(tuple, result))
        return result

    # def create_review(self, review: DBReview, replace: bool = False) -> str:
    # print('backend create_review')
    # poll = self.poll(dict(id=review.pollid))
    # fileid = [f.id for f in poll.pollfiles if f.fname == review.fname]
    # if not fileid: return f'fname {review.fname} not in poll {poll.name}, candidates: {poll.pollfiles}'
    # review.fileid = fileid[0]
    # return self.validate_and_add_to_db(review, replace)

    # def poll_fids(self, id, response_model=dict[str, int]):
    # return {f.fname: f.id for f in self.poll(id).pollfiles}
#
# def poll_file(self,
# id: int,
# request: fastapi.Request,
# response: fastapi.Response,
# shuffle: bool = False,
# trackseen: bool = False):
# poll = self.poll(id)
# files = ordset.OrderedSet(f.fname for f in poll.pollfiles)
# if trackseen:
# seenit = request.cookies.get(f'seenit_poll{id}')
# seenit = set(seenit.split()) if seenit else set()
# files -= seenit
# if not files: return dict(fname=None, next=[])
# idx = random.randrange(len(files)) if shuffle else 0
# if trackseen:
# seenit.add(files[idx])
# response.set_cookie(key=f"seenit_poll{id}", value=' '.join(seenit))
# return dict(fname=files[0], next=files[1:10])

    def reviews_fname(self, fname):
        fname = fname.replace('__DIRSEP__', '/')
        files = self.session.exec(sqlmodel.select(DBPollFile).where(DBPollFile.fname == fname))
        rev = ordset.OrderedSet()
        for f in files:
            rev |= f.reviews
        return list(rev)

    def have_pollfile(self, pollid: int, fname: str):
        poll = self.poll(dict(id=pollid))
        newfname = self.permafname_name(poll, fname)
        return os.path.exists(newfname), newfname

    def permafname_name(self, poll, fname):
        pollname = poll.name.replace(' ', '_').replace('/', '\\')
        path = os.path.join(self.datadir, 'poll', f'{pollname}__{poll.id}', 'reviewed')
        os.makedirs(path, exist_ok=True)
        newfname = os.path.join(path, fname.replace('/', '\\'))
        return newfname

    def create_file_with_content(self, file: DBPollFile):
        assert file.filecontent
        mode = 'wb' if file.permafname.endswith('.bcif') else 'w'
        with open(file.permafname, mode) as out:
            out.write(file.filecontent)

    def create_empty_files(self, files: list[ppp.PollFileSpec]):
        # print('CREATE empty files', len(files))
        for f in files:
            assert not f.filecontent.strip()
            self.session.add(DBPollFile(**f.dict()))
        self.session.commit()

# Autogen getter methods. Yes, this is better than 20 boilerplate functions that must be kept
# in sync. Any name or name suffixed with 's'
# that is in clientmodels, above, will get /name from the server and turn the result(s) into
# the appropriate client model type, list of such types for plural, or None.
for _name, _cls in backend_model.items():

    def make_funcs_forcing_closure_over_name_cls(name=_name, cls=_cls):
        def create(self, model: cls, replace: bool = False) -> str:
            return self.validate_and_add_to_db(model, replace)

        def count(self, kw=None, request: fastapi.Request = None, response_model=list[cls]):
            # print('route', name, cls, kw, request, flush=True)
            if request: return self.select(cls, _count=True, **request.query_params)
            elif kw: return self.select(cls, _count=True, **kw)
            else: return self.select(cls, _count=True)

        def multi(self, kw=None, request: fastapi.Request = None, response_model=list[cls]):
            # print('route', name, cls, kw, request, flush=True)
            if request: return self.select(cls, **request.query_params)
            elif kw: return self.select(cls, **kw)
            else: return self.select(cls)

        def single(self, kw=None, request: fastapi.Request = None, response_model=Optional[cls]):
            # print('route', name, cls, kw, request, flush=True)
            if request: result = self.select(cls, **request.query_params)
            elif kw: result = self.select(cls, **kw)
            else: result = self.select(cls)
            assert len(result) <= 1
            return result[0] if result else None

        return multi, single, count, create

    multi, single, count, create = make_funcs_forcing_closure_over_name_cls()
    setattr(Backend, _name, single)
    setattr(Backend, f'{_name}s', multi)
    setattr(Backend, f'n{_name}s', count)
    if not hasattr(Backend, f'create_{_name}'):
        setattr(Backend, f'create_{_name}', create)

class Server(uvicorn.Server):
    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self, *_):
        print('shutting down server')
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
def run(port, dburl=None, datadir='~/.config/ppp/localserver/data', loglevel='warning', local=False, **kw):
    from rich import print
    datadir = os.path.abspath(os.path.expanduser(datadir))
    dburl = dburl or f'sqlite:///{datadir}/ppp.db'
    if not dburl.count('://'): dburl = f'sqlite:///{dburl}'
    os.makedirs(datadir, exist_ok=True)
    print(f'creating db engine from url: "{dburl}"')
    engine = sqlmodel.create_engine(dburl)
    backend = Backend(engine, datadir)
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
        workers=9,
    )
    server = Server(config=config)
    server.run_in_thread()
    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGINT, server.stop)
    for _ in range(5000):
        if server.started: break
        time.sleep(0.001)
    else:
        raise RuntimeError('server failed to start')
    ppp.set_servermode(True)
    client = ppp.PPPClient(f'127.0.0.1:{port}')
    assert ipd.ppp.get_hack_fixme_global_client()
    ppp.server.defaults.add_defaults(**kw)
    print('server', socket.gethostname())
    return server, backend, client

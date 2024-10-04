import sys
import os
import functools
import random
from datetime import datetime, timedelta
import contextlib
import threading
import time
import socket
import shutil
import operator
from pathlib import Path
from typing import Optional
import ipd
from ipd import ppp
from icecream import ic
import signal
from typing import Union

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

class DuplicateError(Exception):
    def __init__(self, msg, conflict):
        super().__init__(msg)
        self.conflict = conflict

class DBBase:
    def __hash__(self):
        return self.dbkey

    def clear(self, backend):
        return

props_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=list)
attrs_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=dict)

def check_ghost_poll_and_file(backend):
    if not backend.select(DBPoll, dbkey=666):
        backend.session.add(
            DBPoll(name='Ghost Poll',
                   desc='Reviews point here when their poll gets deleted',
                   path='Ghost Dir',
                   user='admin',
                   dbkey=666,
                   ispublic=False))
        backend.session.commit()
    if not backend.select(DBFile, dbkey=666):
        backend.session.add(
            DBFile(name='Ghost File',
                   desc='Reviews point here when their orig file gets deleted. Use the review\'s permafname',
                   fname='Ghost File',
                   user='admin',
                   dbkey=666,
                   polldbkey=666,
                   ispublic=False))
        backend.session.commit()

@profile
class DBPoll(DBBase, ppp.PollSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = props_default()
    attrs: dict[str, Union[str, int, float]] = attrs_default()
    nchain: int = -1
    files: list["DBFile"] = sqlmodel.Relationship(back_populates="poll")
    reviews: list["DBReview"] = sqlmodel.Relationship(back_populates="poll")

    def validated_with_backend(self, backend):
        if conflicts := set(backend.select(DBPoll, name=self.name, dbkeynot=self.dbkey)):
            print('conflicts', [c.name for c in conflicts])
            raise DuplicateError(f'duplicate poll {self.name}', conflicts)
        return self

    def clear(self, backend):
        check_ghost_poll_and_file(backend)
        for r in backend.select(DBReview, polldbkey=self.dbkey):
            r.polldbkey = 666
            r.filedbkey = 666
        backend.session.commit()
        for f in backend.select(DBFile, polldbkey=self.dbkey):
            backend.session.delete(f)

@profile
class DBFile(DBBase, ppp.FileSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = props_default()
    attrs: dict[str, Union[str, int, float]] = attrs_default()
    polldbkey: int = sqlmodel.Field(default=None, foreign_key="dbpoll.dbkey")
    poll: DBPoll = sqlmodel.Relationship(back_populates="files")
    reviews: list['DBReview'] = sqlmodel.Relationship(back_populates='file')

    def validated_with_backend(self, backend):
        # assert os.path.exists(self.fname)
        return self

    @pydantic.validator('fname')
    def valfname(cls, fname):
        return os.path.abspath(fname)

@profile
class DBReview(DBBase, ppp.ReviewSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = props_default()
    attrs: dict[str, Union[str, int, float]] = attrs_default()
    filedbkey: int = sqlmodel.Field(default=None, foreign_key="dbfile.dbkey")
    polldbkey: int = sqlmodel.Field(default=None, foreign_key="dbpoll.dbkey")
    file: DBFile = sqlmodel.Relationship(back_populates='reviews')
    poll: DBPoll = sqlmodel.Relationship(back_populates='reviews')

    def __hash__(self):
        return self.dbkey

    def validated_with_backend(self, backend):
        assert self.file
        assert self.poll

@profile
class DBPymolCMD(DBBase, ppp.PymolCMDSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = props_default()
    attrs: dict[str, Union[str, int, float]] = attrs_default()

    def validated_with_backend(self, backend):
        if conflicts := set(backend.select(DBPymolCMD, name=self.name, dbkeynot=self.dbkey)):
            raise DuplicateError(f'duplicate pymolcmd {self.name}', conflicts)
        return self

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
        self.apply_db_schema_updates()
        self.router = fastapi.APIRouter()
        route = self.router.add_api_route
        route('/', self.root, methods=['GET'])
        route('/create/file', self.create_file_with_content, methods=['POST'])
        route('/create/files', self.create_empty_files, methods=['POST'])
        route('/create/poll', self.create_poll, methods=['POST'])
        route('/create/pymolcmd', self.create_pymolcmd, methods=['POST'])
        route('/create/review', self.create_review, methods=['POST'])
        route('/files', self.files, methods=['GET'])
        route('/getattr/{thing}/{id}/{attr}', self.getattr, methods=['GET'])
        route('/have/file', self.have_file, methods=['POST'])
        route('/pollinfo', self.pollinfo, methods=['GET'])
        route('/polls', self.polls, methods=['GET'])
        route('/poll{dbkey}', self.poll, methods=['GET'])
        route('/poll{dbkey}/fids', self.poll_fids, methods=['GET'])
        route('/poll{dbkey}/fname', self.poll_file, methods=['GET'])
        route('/pymolcmds', self.pymolcmds, methods=['GET'])
        route('/remove/{thing}/{dbkey}', self.remove, methods=['GET'])
        route('/reviews', self.reviews, methods=['GET'])
        route('/reviews/byfname/{fname}', self.reviews_fname, methods=['GET'])
        route('/reviews/file{dbkey}', self.review_for_dbkey, methods=['GET'])
        route('/reviews/poll{dbkey}', self.review_for_dbkey, methods=['GET'])
        route('/review{dbkey}', self.review, methods=['GET'])
        route('/gitstatus/{header}/{footer}', ipd.dev.git_status, methods=['GET'])
        self.app = fastapi.FastAPI()
        # self.app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
        self.app.include_router(self.router)
        ipd.dev.git_status(header='server git status', footer='end', printit=True)

        @self.app.exception_handler(Exception)
        async def validation_exception_handler(request: fastapi.Request, exc: Exception):
            exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
            print('!' * 69)
            print(exc_str)
            print('!' * 69)
            return fastapi.responses.JSONResponse(content=exc_str,
                                                  status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY)

    def _add_db_column(self, table_name, column):
        column_name = column.compile(dialect=self.engine.dialect)
        column_type = column.type.compile(self.engine.dialect)
        cmd = f'ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}'
        print(f'EXEC RAW SQL: "{cmd}"', flush=True)
        self.session.execute(sqlalchemy.text(cmd))
        self.session.commit()

    def apply_db_schema_updates(self):
        tables = 'DBPoll DBFile DBReview DBPymolCMD'.split()
        getcols = lambda table: self.session.execute(sqlalchemy.text(f'select * from {table}')).keys()
        cols = {table: getcols(table.lower()) for table in tables}
        for table in tables:
            cls = globals()[table]
            for name, field in cls.__fields__.items():
                if name not in cols[table]:
                    print('ADD NEW COLUMN',
                          table,
                          name,
                          python_type_to_sqlalchemy_type[field.annotation],
                          flush=True)
                    column = sqlalchemy.Column(name, python_type_to_sqlalchemy_type[field.annotation])
                    self._add_db_column(table.lower(), column)

    def remove(self, thing, dbkey):
        thing = self.select(thing, dbkey=dbkey)
        assert len(thing) == 1
        thing = thing[0]
        thing.clear(self)
        self.session.delete(thing)
        self.session.commit()

    def root(self) -> None:
        return dict(msg='Hello World')

    def getattr(self, thing, id, attr):
        thingtype = globals()[f'DB{thing.title()}']
        thing = next(self.session.exec(sqlmodel.select(thingtype).where(thingtype.dbkey == id)))
        thingattr = getattr(thing, attr)
        return thingattr

    def select(self, type_, **kw):
        if isinstance(type_, str):
            type_ = dict(poll=DBPoll, file=DBFile, review=DBReview, pymolcmd=DBPymolCMD)[type_]
        statement = sqlmodel.select(type_)
        for k, v in kw.items():
            op = operator.eq
            if k.endswith('not'): k, op = k[:-3], operator.ne
            if v is not None:
                # print('select where', type_, k, v)
                statement = statement.where(op(getattr(type_, k), v))
        return list(self.session.exec(statement))

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
                # print('DB ADDED', thing)
            except AssertionError as e:
                self.ssesion.delete(thing)
                return f'error in validate_and_add_to_db: {e}'
        except DuplicateError as e:
            if replace:
                for oldthing in e.conflict:
                    print('DB DELETE', type(oldthing), oldthing.dbkey)
                    oldthing.clear(self)
                    self.session.delete(oldthing)
                # print('DB ADDED', thing)
            else:
                thing.clear(self)
                self.session.delete(thing)
                result = 'duplicate'
        self.session.commit()
        return result

    def poll(self, dbkey: int, response_model=Optional[DBPoll]):
        poll = self.select(DBPoll, dbkey=dbkey)
        return poll[0] if poll else None

    def file(self, dbkey: int, response_model=Optional[DBFile]):
        file = self.select(DBFile, dbkey=dbkey)
        return file[0] if file else None

    def review(self, dbkey: int, response_model=Optional[DBReview]):
        review = self.select(DBReview, dbkey=dbkey)
        return review[0] if review else None

    def pymolcmd(self, dbkey: int, response_model=Optional[DBPymolCMD]):
        cmd = self.select(DBPymolCMD, dbkey=dbkey)
        return cmd[0] if cmd else None

    def polls(self, dbkey: int = None, name=None, response_model=list[DBPoll]):
        return self.select(DBPoll, dbkey=dbkey, name=name)

    def files(self, dbkey: int = None, response_model=list[DBFile]):
        return self.select(DBFile, dbkey=dbkey)

    def reviews(self, dbkey: int = None, name=None, response_model=list[DBReview]):
        return self.select(DBReview, dbkey=dbkey, name=name)

    def pymolcmds(self, dbkey: int = None, name=None, response_model=list[DBPymolCMD]):
        return self.select(DBPymolCMD, dbkey=None, name=None)

    def pollinfo(self, user=None):
        print(f'server pollinfo {user}')
        query = f'SELECT dbkey,name,dbpoll.user,"desc",sym,ligand,nchain FROM dbpoll WHERE ispublic OR dbpoll.user=\'{user}\';'
        if not user or user == 'admin':
            query = 'SELECT dbkey,name,dbpoll.user,"desc",sym,ligand,nchain FROM dbpoll'
        result = self.session.execute(sqlalchemy.text(query)).fetchall()
        return list(map(tuple, result))

    def create_poll(self, poll: DBPoll, replace: bool = False) -> str:
        print('create_poll', poll)
        return self.validate_and_add_to_db(poll, replace)

    def create_review(self, review: DBReview, replace: bool = False) -> str:
        # print('backend create_review')
        poll = self.poll(review.polldbkey)
        filedbkey = [f.dbkey for f in poll.files if f.fname == review.fname]
        if not filedbkey: return f'fname {review.fname} not in poll {poll.name}, candidates: {poll.files}'
        review.filedbkey = filedbkey[0]
        return self.validate_and_add_to_db(review, replace)

    def create_pymolcmd(self, pymolcmd: DBPymolCMD, replace: bool = False) -> str:
        # print('backend create_pymolcmd replace:', replace)
        return self.validate_and_add_to_db(pymolcmd, replace)

    def poll_fids(self, dbkey, response_model=dict[str, int]):
        return {f.fname: f.dbkey for f in self.poll(dbkey).files}

    def poll_file(self,
                  dbkey: int,
                  request: fastapi.Request,
                  response: fastapi.Response,
                  shuffle: bool = False,
                  trackseen: bool = False):
        poll = self.poll(dbkey)
        files = ordset.OrderedSet(f.fname for f in poll.files)
        if trackseen:
            seenit = request.cookies.get(f'seenit_poll{dbkey}')
            seenit = set(seenit.split()) if seenit else set()
            files -= seenit
        if not files: return dict(fname=None, next=[])
        idx = random.randrange(len(files)) if shuffle else 0
        if trackseen:
            seenit.add(files[idx])
            response.set_cookie(key=f"seenit_poll{dbkey}", value=' '.join(seenit))
        return dict(fname=files[0], next=files[1:10])

    def review_for_dbkey(self, dbkey):
        return self.poll(dbkey).reviews

    def review_for_dbkey(self, dbkey):
        return self.file(dbkey).reviews

    def reviews_fname(self, fname):
        fname = fname.replace('__DIRSEP__', '/')
        files = self.session.exec(sqlmodel.select(DBFile).where(DBFile.fname == fname))
        rev = ordset.OrderedSet()
        for f in files:
            rev |= f.reviews
        return list(rev)

    def have_file(self, file: DBFile):
        poll = self.poll(file.polldbkey)
        newfname = self.permafname_name(poll, file.fname)
        return os.path.exists(newfname), newfname

    def permafname_name(self, poll, fname):
        pollname = poll.name.replace(' ', '_').replace('/', '\\')
        path = os.path.join(self.datadir, 'poll', f'{pollname}__{poll.dbkey}', 'reviewed')
        os.makedirs(path, exist_ok=True)
        newfname = os.path.join(path, fname.replace('/', '\\'))
        return newfname

    def create_file_with_content(self, file: DBFile):
        assert file.filecontent
        mode = 'wb' if file.permafname.endswith('.bcif') else 'w'
        with open(file.permafname, mode) as out:
            out.write(file.filecontent)

    def create_empty_files(self, files: list[ppp.FileSpec]):
        # print('CREATE empty files', len(files))
        for f in files:
            assert not f.filecontent.strip()
            self.session.add(DBFile(**f.dict()))
        self.session.commit()

class Server(uvicorn.Server):
    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self, sig, frame):
        print('shutting down server')
        self.should_exit = True
        self.thread.join()
        sys.exit()

def pymol_launch():
    stdout = sys.stdout
    stderr = sys.stderr
    pymol.finish_launching(['pymol', '-xiqckK'])
    sys.stdout = stdout
    sys.stderr = stderr
    pymol.cmd.set('suspend_updates', 'on')

@profile
def run(port, dburl=None, datadir='~/.config/ppp/localserver/data', loglevel='info', local=False, **kw):
    from fastapi.middleware.gzip import GZipMiddleware
    import pymol
    ppp.SERVER = True
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
    ppp.set_server(True)
    check_ghost_poll_and_file(backend)
    ppp.defaults.add_defaults(f'127.0.0.1:{port}', **kw)
    print('server', socket.gethostname())
    return server, backend

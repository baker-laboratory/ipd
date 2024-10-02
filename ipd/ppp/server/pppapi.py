import sys
import os
import functools
import random
from datetime import datetime, timedelta
import contextlib
import threading
import time
import shutil
import operator
from pathlib import Path
from typing import Optional
import ipd
from ipd import ppp
from icecream import ic
import uvicorn
import signal
from fastapi.middleware.gzip import GZipMiddleware

fastapi = ipd.lazyimport('fastapi', pip=True)
pydantic = ipd.lazyimport('pydantic', pip=True)
sqlmodel = ipd.lazyimport('sqlmodel', pip=True)
sqlalchemy = ipd.lazyimport('sqlalchemy', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
pymol = ipd.lazyimport('pymol', 'pymol-bundle', mamba=True, channels='-c schrodinger')

SESSION = None

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

@profile
class DBPoll(DBBase, ppp.PollSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    nchain: int = -1
    props: list[str] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType), default_factory=list)
    attrs: dict[str, str | int | float] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType),
                                                         default_factory=dict)
    files: list["DBFile"] = sqlmodel.Relationship(back_populates="poll")
    reviews: list["DBReview"] = sqlmodel.Relationship(back_populates="poll")

    def populate_files(self):
        assert os.path.isdir(self.path)
        filt = lambda s: not s.startswith('_') and s.endswith(ppp.STRUCTURE_FILE_SUFFIX)
        if fnames := filter(filt, os.listdir(self.path)):
            for fname in fnames:
                assert self.dbkey
                file = DBFile(polldbkey=self.dbkey, fname=os.path.join(self.path, fname))
                SESSION.add(file.validated_with_backend(None))
            return True
        else:
            self._errors += 'no valid filenames in dir {self.path}'
            return False

    def validated_with_backend(self, backend):
        if not self.populate_files(): return self
        if conflicts := set(backend.select(DBPoll, name=self.name, dbkeynot=self.dbkey)):
            print('conflicts', [c.name for c in conflicts])
            raise DuplicateError(f'duplicate poll {self.name}', conflicts)
        return self

    def clear(self, backend):
        for f in backend.select(DBFile, polldbkey=self.dbkey):
            backend.session.delete(f)

@profile
class DBFile(DBBase, ppp.FileSpec, sqlmodel.SQLModel, table=True):
    dbkey: int | None = sqlmodel.Field(default=None, primary_key=True)
    polldbkey: int = sqlmodel.Field(default=None, foreign_key="dbpoll.dbkey")
    poll: DBPoll = sqlmodel.Relationship(back_populates="files")
    reviews: list['DBReview'] = sqlmodel.Relationship(back_populates='file')
    props: list[str] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType), default_factory=list)
    attrs: dict[str, str | int | float] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType),
                                                         default_factory=dict)

    def validated_with_backend(self, backend):
        assert os.path.exists(self.fname)
        return self

    @pydantic.validator('fname')
    def valfname(cls, fname):
        return os.path.abspath(fname)

@profile
class DBReview(DBBase, ppp.ReviewSpec, sqlmodel.SQLModel, table=True):
    dbkey: int | None = sqlmodel.Field(default=None, primary_key=True)
    filedbkey: int = sqlmodel.Field(default=None, foreign_key="dbfile.dbkey")
    polldbkey: int = sqlmodel.Field(default=None, foreign_key="dbpoll.dbkey")
    file: DBFile = sqlmodel.Relationship(back_populates='reviews')
    poll: DBPoll = sqlmodel.Relationship(back_populates='reviews')
    props: list[str] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType), default_factory=list)
    attrs: dict[str, str | int | float] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType),
                                                         default_factory=dict)

    def __hash__(self):
        return self.dbkey

    def validated_with_backend(self, backend):
        assert self.file
        assert self.poll

@profile
class DBPymolCMD(DBBase, ppp.PymolCMDSpec, sqlmodel.SQLModel, table=True):
    dbkey: int | None = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType), default_factory=list)
    attrs: dict[str, str | int | float] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType),
                                                         default_factory=dict)

    def validated_with_backend(self, backend):
        if conflicts := set(backend.select(DBPymolCMD, name=self.name, dbkeynot=self.dbkey)):
            raise DuplicateError(f'duplicate pymolcmd {self.name}', conflicts)
        return self

@profile
class Backend:
    def __init__(self, engine, datadir):
        self.engine = engine
        self.datadir = datadir
        self.session = sqlmodel.Session(engine)
        global SESSION
        assert not SESSION
        SESSION = self.session
        sqlmodel.SQLModel.metadata.create_all(self.engine)
        self.router = fastapi.APIRouter()
        self.router.add_api_route('/', self.root, methods=['GET'])
        self.router.add_api_route('/polls', self.polls, methods=['GET'])
        self.router.add_api_route('/pollinfo', self.pollinfo, methods=['GET'])
        self.router.add_api_route('/reviews', self.reviews, methods=['GET'])
        self.router.add_api_route('/files', self.files, methods=['GET'])
        self.router.add_api_route('/pymolcmds', self.pymolcmds, methods=['GET'])
        self.router.add_api_route('/poll{dbkey}', self.poll, methods=['GET'])
        self.router.add_api_route('/poll{dbkey}/fname', self.poll_file, methods=['GET'])
        self.router.add_api_route('/poll{dbkey}/fids', self.poll_fids, methods=['GET'])
        self.router.add_api_route('/reviews/poll{dbkey}', self.review_for_dbkey, methods=['GET'])
        self.router.add_api_route('/reviews/file{dbkey}', self.review_for_dbkey, methods=['GET'])
        self.router.add_api_route('/reviews/byfname/{fname}', self.reviews_fname, methods=['GET'])
        self.router.add_api_route('/review{dbkey}', self.review, methods=['GET'])
        self.router.add_api_route('/create/poll', self.create_poll, methods=['POST'])
        self.router.add_api_route('/create/review', self.create_review, methods=['POST'])
        self.router.add_api_route('/create/pymolcmd', self.create_pymolcmd, methods=['POST'])
        self.router.add_api_route('/have/file', self.have_file, methods=['POST'])
        self.router.add_api_route('/create/file', self.create_file, methods=['POST'])
        self.router.add_api_route('/getattr/{thing}/{id}/{attr}', self.getattr, methods=['GET'])
        self.app = fastapi.FastAPI()
        # self.app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
        self.app.include_router(self.router)

        @self.app.exception_handler(Exception)
        async def validation_exception_handler(request: fastapi.Request, exc: Exception):
            print('!' * 100)
            exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
            return fastapi.responses.JSONResponse(content=exc_str,
                                                  status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY)

    def root(self) -> None:
        return dict(msg='Hello World')

    def getattr(self, thing, id, attr):
        thingtype = globals()[f'DB{thing.title()}']
        thing = next(self.session.exec(sqlmodel.select(thingtype).where(thingtype.dbkey == id)))
        thingattr = getattr(thing, attr)
        return thingattr

    def select(self, type_, **kw):
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
                print('DB ADDED', thing)
            except AssertionError as e:
                self.ssesion.delete(thing)
                return f'error in validate_and_add_to_db: {e}'
        except DuplicateError as e:
            if replace:
                for oldthing in e.conflict:
                    print('DB DELETE', type(oldthing), oldthing.dbkey)
                    oldthing.clear(self)
                    self.session.delete(oldthing)
                print('DB ADDED', thing)
            else:
                thing.clear(self)
                self.session.delete(thing)
                result = 'duplicate'
        self.session.commit()
        return result

    def poll(self, dbkey: int, response_model=DBPoll | None):
        poll = self.select(DBPoll, dbkey=dbkey)
        return poll[0] if poll else None

    def file(self, dbkey: int, response_model=DBFile | None):
        file = self.select(DBFile, dbkey=dbkey)
        return file[0] if file else None

    def review(self, dbkey: int, response_model=DBReview | None):
        review = self.select(DBReview, dbkey=dbkey)
        return review[0] if review else None

    def pymolcmd(self, dbkey: int, response_model=DBPymolCMD | None):
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
        query = f'SELECT dbkey,name,user,"desc",sym,ligand,nchain FROM dbpoll WHERE ispublic OR user=\'{user}\';'
        if not user: query = 'SELECT dbkey,name,user,"desc",sym,ligand FROM dbpoll'
        result = self.session.execute(sqlalchemy.text(query)).fetchall()
        return list(map(tuple, result))

    def create_poll(self, poll: DBPoll, replace: bool = False) -> str:
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
        print(poll)
        print(self.polls())
        newfname = self.permafname_name(poll, file.fname)
        return os.path.exists(newfname), newfname

    def permafname_name(self, poll, fname):
        pollname = poll.name.replace(' ', '_').replace('/', '\\')
        path = os.path.join(self.datadir, 'poll', f'{pollname}__{poll.dbkey}', 'reviewed')
        os.makedirs(path, exist_ok=True)
        newfname = os.path.join(path, fname.replace('/', '\\'))
        return newfname

    def create_file(self, file: DBFile):
        print('MAKEpermafname', file.permafname)
        assert file.filecontent
        mode = 'wb' if file.permafname.endswith('.bcif') else 'w'
        with open(file.permafname, mode) as out:
            out.write(file.filecontent)

class Server(uvicorn.Server):
    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self, sig, frame):
        print('shutting down server')
        self.should_exit = True
        self.thread.join()
        sys.exit()

@profile
def run(port, dburl=None, datadir='~/.config/ppp/localserver/data', loglevel='info', **kw):
    import pymol
    datadir = os.path.abspath(os.path.expanduser(datadir))
    dburl = dburl or f'sqlite:///{datadir}/ppp.db'
    if not dburl.count('://'): dburl = f'sqlite:///{dburl}'
    os.makedirs(datadir, exist_ok=True)
    print(f'creating db engine from url: "{dburl}"')
    engine = sqlmodel.create_engine(dburl)
    backend = Backend(engine, datadir)
    backend.app.mount("/ppp", backend.app)
    pymol.pymol_argv = ['pymol', '-qckK']
    pymol.finish_launching()
    config = uvicorn.Config(backend.app, host="0.0.0.0", port=port, log_level=loglevel)
    server = Server(config=config)
    server.run_in_thread()
    signal.signal(signal.SIGINT, server.stop)
    ppp.defaults.add_defaults(f'0.0.0.0:{port}', **kw)
    return server, backend

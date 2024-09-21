import os
import functools
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import ipd
from icecream import ic

fastapi = ipd.lazyimport('fastapi', pip=True)
pydantic = ipd.lazyimport('pydantic', pip=True)
sqlmodel = ipd.lazyimport('sqlmodel', pip=True)
sqlalchemy = ipd.lazyimport('sqlalchemy', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
pymol = ipd.lazyimport('pymol', 'pymol-bundle', mamba=True, channels='-c schrodinger')

SESSION = None

class DBPoll(ipd.ppp.PollSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType), default_factory=list)
    attrs: dict[str, str | int | float] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType),
                                                         default_factory=dict)
    files: list["DBFile"] = sqlmodel.Relationship(back_populates="poll")
    reviews: list["DBReview"] = sqlmodel.Relationship(back_populates="poll")

    def replace_dir_with_pdblist(self, datadir, suffix, recurse=False):
        # sourcery skip: extract-method
        path = Path(self.path)
        if path.is_dir:
            newpath = f'{datadir}/poll/{self.dbkey}.filelist'
            os.makedirs(os.path.dirname(newpath), exist_ok=True)
            with open(newpath, 'w') as out:
                if recurse:
                    for root, dirs, files in os.walk(path):
                        for fname in (_ for _ in files if _.endswith(suffix) and not _.startswith('_')):
                            out.write(os.path.abspath(os.path.join(path, root, fname)) + os.linesep)
                else:
                    for fname in [_ for _ in os.listdir(path) if _.endswith(suffix) and not _.startswith('_')]:
                        out.write(os.path.join(path, fname) + os.linesep)
            self.path = newpath
            assert os.path.getsize(newpath)
            for fname in open(newpath):
                assert os.path.exists(fname.strip()), fname
        else:
            with open(self.path) as inp:
                fnames = [os.path.realpath(_) for _ in inp]
            with open(self.path, 'w') as out:
                out.write(os.linesep.join(fnames) + os.linesep)

    def populate_files(self):
        with open(self.path) as inp:
            for line in inp:
                file = DBFile(polldbkey=self.dbkey, fname=line.strip())
                SESSION.add(file.validated())

class DBFile(ipd.ppp.FileSpec, sqlmodel.SQLModel, table=True):
    dbkey: int | None = sqlmodel.Field(default=None, primary_key=True)
    polldbkey: int = sqlmodel.Field(default=None, foreign_key="dbpoll.dbkey")
    poll: DBPoll = sqlmodel.Relationship(back_populates="files")
    reviews: list['DBReview'] = sqlmodel.Relationship(back_populates='file')
    props: list[str] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType), default_factory=list)
    attrs: dict[str, str | int | float] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType),
                                                         default_factory=dict)

    def validated(self):
        assert os.path.exists(self.fname)
        return self

class DBReview(ipd.ppp.ReviewSpec, sqlmodel.SQLModel, table=True):
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

    def validated(self):
        assert self.file
        assert self.poll
        return self

class DBPymolCMD(ipd.ppp.PymolCMDSpec, sqlmodel.SQLModel, table=True):
    dbkey: int | None = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType), default_factory=list)
    attrs: dict[str, str | int | float] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType),
                                                         default_factory=dict)

    def validated(self):
        return self

class Server:
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
        self.router.add_api_route('/getattr/{thing}/{id}/{attr}', self.getattr, methods=['GET'])
        self.app = fastapi.FastAPI()
        self.app.include_router(self.router)
        self.suffix = tuple('.pdb .pdb.gz .cif .bcif'.split())

    def root(self) -> None:
        return dict(msg='Hello World')

    def getattr(self, thing, id, attr):
        thingtype = globals()[f'DB{thing.title()}']
        thing = next(self.session.exec(sqlmodel.select(thingtype).where(thingtype.dbkey == id)))
        thingattr = getattr(thing, attr)
        return thingattr

    def fix_date(self, x):
        if hasattr(x, 'datecreated') and isinstance(x.datecreated, str):
            x.datecreated = datetime.strptime(x.datecreated, ipd.ppp.DTFMT)
        if hasattr(x, 'enddate') and isinstance(x.enddate, str):
            x.enddate = datetime.strptime(x.enddate, ipd.ppp.DTFMT)

    def create_poll(self, poll: DBPoll) -> None:
        # print(f'SERVER ADD_POL {poll.name} {poll.path}')
        self.fix_date(poll)
        self.session.add(poll)
        self.session.commit()  # sets dbkey
        poll.replace_dir_with_pdblist(self.datadir, self.suffix)
        poll.populate_files()
        self.session.commit()

    def poll(self, dbkey, response_model=DBPoll):
        poll = list(self.session.exec(sqlmodel.select(DBPoll).where(DBPoll.dbkey == dbkey)))
        return poll[0] if poll else None

    def polls(self, response_model=list[DBPoll]):
        return list(self.session.exec(sqlmodel.select(DBPoll)))

    def files(self, response_model=list[DBFile]):
        return list(self.session.exec(sqlmodel.select(DBFile)))

    def pymolcmds(self, response_model=list[DBPymolCMD]):
        return list(self.session.exec(sqlmodel.select(DBPymolCMD)))

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

    def file(self, dbkey):
        file = list(self.session.exec(sqlmodel.select(DBFile).where(DBFile.dbkey == dbkey)))
        return file[0] if file else None

    def reviews(self):
        return list(self.session.exec(sqlmodel.select(DBReview)))

    def review(self, dbkey):
        return self.session.exec(sqlmodel.select(DBReview).where(DBReview.dbkey == dbkey))

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

    def create_review(self, review: DBReview):
        poll = self.poll(review.polldbkey)
        filedbkey = [f.dbkey for f in poll.files if f.fname == review.fname]
        if not filedbkey:
            raise ValueError(f'fname {review.fname} not in poll {poll.name}, candidates: {poll.files}')
        assert len(filedbkey) == 1
        review.filedbkey = filedbkey[0]
        self.validate_and_add_to_db(review)

    def create_pymolcmd(self, pymolcmd: DBPymolCMD):
        self.validate_and_add_to_db(pymolcmd)

    def validate_and_add_to_db(self, thing):
        self.fix_date(thing)
        self.session.add(thing)
        self.session.commit()
        thing.validated()
        self.session.commit()

def run(port, datadir, log='info'):
    import uvicorn
    os.makedirs(datadir, exist_ok=True)
    engine = sqlmodel.create_engine(f'sqlite:///{datadir}/ppp.db')
    server = Server(engine, datadir)
    server.app.mount("/ppp", server.app)  # your app routes will now be /app/{your-route-here}
    return uvicorn.run(server.app, host="0.0.0.0", port=port, log_level=log)

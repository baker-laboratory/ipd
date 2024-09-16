import os
import functools
import random
from datetime import datetime, timedelta
import fastapi
from pathlib import Path
from typing import Optional
from pydantic import FilePath, DirectoryPath, validator
import sqlmodel
import sqlalchemy
import pydantic
import ipd
from ordered_set import OrderedSet as ordset
from icecream import ic

SESSION = None

class ServerPoll(ipd.ppp.PollUpload, sqlmodel.SQLModel, table=True):
    pollid: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    public: bool = sqlmodel.Field(default=False)
    telemetry: bool = sqlmodel.Field(default=False)
    start: datetime = sqlmodel.Field(default_factory=datetime.now)
    end: datetime = sqlmodel.Field(default_factory=lambda: datetime.now() + timedelta(days=3))
    props: list[str] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType), default_factory=list)
    attrs: dict[str, str | int] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType),
                                                 default_factory=dict)
    files: list["File"] = sqlmodel.Relationship(back_populates="poll")
    reviews: list["ServerReview"] = sqlmodel.Relationship(back_populates="poll")

    # TODO add fnames to database
    def replace_dir_with_pdblist(self, datadir, suffix, recurse=False):
        # sourcery skip: extract-method
        path = Path(self.path)
        if path.is_dir:
            newpath = f'{datadir}/poll/{self.pollid}.filelist'
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
                file = File(pollid=self.pollid, fname=line.strip())
                SESSION.add(file.validated())

class File(sqlmodel.SQLModel, table=True):
    fileid: int | None = sqlmodel.Field(default=None, primary_key=True)
    pollid: int | None = sqlmodel.Field(default=None, foreign_key="serverpoll.pollid")
    poll: ServerPoll | None = sqlmodel.Relationship(back_populates="files")
    reviews: list['ServerReview'] = sqlmodel.Relationship(back_populates='file')
    fname: str

    def validated(self):
        assert os.path.exists(self.fname)
        return self

class ServerReview(ipd.ppp.ReviewUpload, sqlmodel.SQLModel, table=True):
    reviewid: int | None = sqlmodel.Field(default=None, primary_key=True)
    fileid: int | None = sqlmodel.Field(default=None, foreign_key="file.fileid")
    pollid: int | None = sqlmodel.Field(default=None, foreign_key="serverpoll.pollid")
    file: File | None = sqlmodel.Relationship(back_populates='reviews')
    poll: ServerPoll | None = sqlmodel.Relationship(back_populates='reviews')
    attrs: dict[str, str | int | float] = sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.PickleType),
                                                         default_factory=dict)
    def __hash__(self):
        return self.reviewid

    def validated(self):
        assert self.file
        assert self.poll
        print(self.fname)
        print(self.file)
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
        self.router.add_api_route("/", self.root, methods=["GET"])
        self.router.add_api_route("/polls", self.poll, methods=["GET"])
        self.router.add_api_route("/poll{pollid}", self.poll, methods=["GET"])
        self.router.add_api_route("/poll", self.add_poll, methods=["POST"])
        self.router.add_api_route("/poll{pollid}/fname", self.poll_file, methods=["GET"])
        self.router.add_api_route("/poll{pollid}/fids", self.poll_fids, methods=["GET"])
        self.router.add_api_route("/reviews", self.reviews, methods=["GET"])
        self.router.add_api_route("/reviews/poll{pollid}", self.review_for_pollid, methods=["GET"])
        self.router.add_api_route("/reviews/file{fileid}", self.review_for_fileid, methods=["GET"])
        self.router.add_api_route("/reviews/byfname/{fname}", self.reviews_fname, methods=["GET"])
        self.router.add_api_route("/review{reviewid}", self.review, methods=["GET"])
        self.router.add_api_route("/poll{pollid}/review", self.post_review, methods=["POST"])

        self.app = fastapi.FastAPI()
        self.app.include_router(self.router)
        self.suffix = tuple('.pdb .pdb.gz .cif .bcif'.split())

    def root(self) -> None:
        return dict(msg='Hello World')

    def add_poll(self, poll: ServerPoll) -> None:
        print(f'SERVER ADD_POL {poll.name} {poll.path}')
        dtfmt = "%Y-%m-%dT%H:%M:%S.%f"
        if isinstance(poll.start, str): poll.start = datetime.strptime(poll.start, dtfmt)
        if isinstance(poll.end, str): poll.end = datetime.strptime(poll.end, dtfmt)
        self.session.add(poll)
        self.session.commit()  # sets pollid
        poll.replace_dir_with_pdblist(self.datadir, self.suffix)
        poll.populate_files()
        self.session.commit()

    def poll(self, pollid, response_model=ServerPoll):
        poll = list(self.session.exec(sqlmodel.select(ServerPoll).where(ServerPoll.pollid == pollid)))
        return poll[0] if poll else None

    def polls(self, response_model=list[ServerPoll]):
        polls = self.session.exec(sqlmodel.select(ServerPoll))
        return list(polls)

    def poll_fids(self, pollid, response_model=dict[str, int]):
        return {f.fname: f.fileid for f in self.poll(pollid).files}

    def poll_file(self,
                  pollid: int,
                  request: fastapi.Request,
                  response: fastapi.Response,
                  shuffle: bool = False,
                  trackseen: bool = False):
        poll = self.poll(pollid)
        files = ordset(f.fname for f in poll.files)
        if trackseen:
            seenit = request.cookies.get(f'seenit_poll{pollid}')
            seenit = set(seenit.split()) if seenit else set()
            files -= seenit
        if not files: return dict(fname=None, next=[])
        idx = random.randrange(len(files)) if shuffle else 0
        if trackseen:
            seenit.add(files[idx])
            response.set_cookie(key=f"seenit_poll{pollid}", value=' '.join(seenit))
        return dict(fname=files[0], next=files[1:10])

    def file(self, fileid):
        file = list(self.session.exec(sqlmodel.select(File).where(File.fileid==fileid)))
        return file[0] if file else None

    def reviews(self):
        return list(self.session.exec(sqlmodel.select(ServerReview)))

    def review(self, reviewid):
        return self.session.exec(sqlmodel.select(ServerReview).where(ServerReview.reviewid == reviewid))

    def review_for_pollid(self, pollid):
        return self.poll(pollid).reviews

    def review_for_fileid(self, fileid):
        return self.file(fileid).reviews

    def reviews_fname(self, fname):
        fname = fname.replace('__DIRSEP__', '/')
        files = self.session.exec(sqlmodel.select(File).where(File.fname == fname))
        rev = ordset()
        for f in files:
            rev |= f.reviews
        return list(rev)

    def post_review(self, pollid, review: ServerReview):
        poll = self.poll(pollid)
        review.pollid = pollid
        fileid = [f.fileid for f in poll.files if f.fname==review.fname]
        if not fileid: raise ValueError(f'fname {review.fname} not in poll {poll.name}, candidates: {poll.files}')
        assert len(fileid) == 1
        review.fileid = fileid[0]
        self.session.add(review)
        self.session.commit()
        review.validated()
        self.session.commit()

def run(port, datadir, log='info'):
    import uvicorn
    os.makedirs(datadir, exist_ok=True)
    engine = sqlmodel.create_engine(f'sqlite:///{datadir}/ppp.db')
    server = Server(engine, datadir)
    server.app.mount("/ppp", server.app)  # your app routes will now be /app/{your-route-here}
    return uvicorn.run(server.app, host="0.0.0.0", port=port, log_level=log)

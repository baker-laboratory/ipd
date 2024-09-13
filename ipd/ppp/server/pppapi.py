import os
import functools
import random
import fastapi
from ordered_set import OrderedSet as ordset
from pathlib import Path
from typing import Optional
from pydantic import FilePath, DirectoryPath, validator
import sqlmodel
from icecream import ic

class Poll(sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    name: str
    desc: str
    path: str
    public: bool = sqlmodel.Field(default=False)

    def __hash__(self):
        return hash(self.path)

    def replace_dir_with_pdblist(self, datadir, suffix):
        path = Path(self.path)
        if path.is_dir:
            newpath = f'{datadir}/poll/{self.id}.pdblist'
            os.makedirs(os.path.dirname(newpath), exist_ok=True)
            with open(newpath, 'w') as out:
                for root, dirs, files in os.walk(path):
                    for name in (_ for _ in files if _.endswith(suffix)):
                        out.write(os.path.join(root, name) + os.linesep)
            self.path = newpath
        else:
            with open(self.path) as inp:
                fnames = [os.path.realpath(_) for _ in inp]
            with open(self.path, 'w') as out:
                out.write(os.linesep.join(fnames) + os.linesep)

    @property
    @functools.lru_cache
    def files(self):
        with open(self.path) as inp:
            return ordset([_.strip() for _ in inp.readlines()])

class Review(sqlmodel.SQLModel, table=True):
    reviewid: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    pollid: Optional[int] = None
    file: str
    user: str
    grade: str
    data: str




class PPPServer:
    def __init__(self, engine, datadir):
        self.engine = engine
        self.datadir = datadir
        self.session = sqlmodel.Session(engine)
        sqlmodel.SQLModel.metadata.create_all(self.engine)
        self.router = fastapi.APIRouter()
        self.router.add_api_route("/", self.root, methods=["GET"])
        self.router.add_api_route("/poll", self.poll, methods=["GET"])
        self.router.add_api_route("/poll{pollid}", self.poll, methods=["GET"])
        self.router.add_api_route("/poll", self.add_poll, methods=["POST"])
        self.router.add_api_route("/poll{pollid}/file", self.poll_file, methods=["GET"])
        self.router.add_api_route("/poll{pollid}/review", self.poll_review, methods=["POST"])
        self.app = fastapi.FastAPI()
        self.app.include_router(self.router)
        self.suffix = tuple('.pdb .pdb.gz .cif .bcif'.split())

    def root(self) -> None:
        return dict(msg='Hello World')

    def add_poll(self, poll: Poll) -> None:
        self.session.add(poll)
        self.session.commit()  # sets pollid
        poll.replace_dir_with_pdblist(self.datadir, self.suffix)
        self.session.commit()

    def poll(self, pollid=None) -> list[Poll]:
        if pollid is None: polls = self.session.exec(sqlmodel.select(Poll))
        else: polls = self.session.exec(sqlmodel.select(Poll).where(Poll.pollid == pollid))
        return list(polls)

    def poll_file(self, pollid: int, request: fastapi.Request, response: fastapi.Response, shuffle: bool = False):
        seenit = request.cookies.get(f'seenit_poll{pollid}')
        # ic(dir(request.cookies.get(f'seenit_poll{pollid}')))
        seenit = set(seenit.split()) if seenit else set()
        poll = self.poll(pollid)
        assert len(poll) == 1
        files = poll[0].files
        files -= seenit
        if not files: return dict(file=None, next=[])
        idx = random.randrange(len(files)) if shuffle else 0
        seenit.add(files[idx])
        response.set_cookie(key=f"seenit_poll{pollid}", value=' '.join(seenit))
        return dict(file=files[0], next=files[1:10])

    def poll_review(self, Review: Review):
        self.session.add(review)
        self.session.commit

import sys
import os
from datetime import datetime
from typing import Optional
import ipd
from ipd import ppp
from typing import Union

pydantic = ipd.lazyimport('pydantic', pip=True)
sqlmodel = ipd.lazyimport('sqlmodel', pip=True)
sqlalchemy = ipd.lazyimport('sqlalchemy', pip=True)

class DBBase:
    def __hash__(self):
        return self.dbkey

    def clear(self, backend):
        return

props_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=list)
attrs_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=dict)

class DuplicateError(Exception):
    def __init__(self, msg, conflict):
        super().__init__(msg)
        self.conflict = conflict

class DBPoll(DBBase, ppp.PollSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = props_default()
    attrs: dict[str, Union[str, int, float]] = attrs_default()
    nchain: int = -1
    files: list["DBFile"] = sqlmodel.Relationship(back_populates="poll")
    reviews: list["DBReview"] = sqlmodel.Relationship(back_populates="poll")
    workflowdbkey: int = sqlmodel.Field(foreign_key="workflow.dbkey")
    workflow: DBWorkFlow = sqlmodel.Relationship(back_populates='polls')

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

class DBPymolCMD(DBBase, ppp.PymolCMDSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = props_default()
    attrs: dict[str, Union[str, int, float]] = attrs_default()

    # steps many to many

    def validated_with_backend(self, backend):
        if conflicts := set(backend.select(DBPymolCMD, name=self.name, dbkeynot=self.dbkey)):
            raise DuplicateError(f'duplicate pymolcmd {self.name}', conflicts)
        return self

class DBFlowStep(DBBase, ppp.FlowStepSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = props_default()
    attrs: dict[str, Union[str, int, float]] = attrs_default()
    workflowdbkey: int = sqlmodel.Field(foreign_key="workflow.dbkey")
    workflow: DBWorkFlow = sqlmodel.Relationship(back_populates='steps')
    # cmds many to many

class DBWorkflow(DBBase, ppp.WorkflowSpec, sqlmodel.SQLModel, table=True):
    dbkey: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = props_default()
    attrs: dict[str, Union[str, int, float]] = attrs_default()
    steps: list["DBFlowStep"] = sqlmodel.Relationship(back_populates="workflow")
    polls: list["DBPoll"] = sqlmodel.Relationship(back_populates="workflow")

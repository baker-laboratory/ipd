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
    # class Config:
    # arbitrary_types_allowed = True

    def __hash__(self):
        return self.id

    def clear(self, backend):
        return

list_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=list)
dict_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=dict)

class DuplicateError(Exception):
    def __init__(self, msg, conflict):
        super().__init__(msg)
        self.conflict = conflict

class DBPymolCMDFlowStepLink(sqlmodel.SQLModel, table=True):
    pymolcmdid: int | None = sqlmodel.Field(default=None, foreign_key='dbpymolcmd.id', primary_key=True)
    flowstepid: int | None = sqlmodel.Field(default=None, foreign_key='dbflowstep.id', primary_key=True)

# class DBPollWorkflowLink(sqlmodel.SQLModel, table=True):
#     pollid: int | None = Field(default=None, foreign_key='dbpoll.id', primary_key=True)
#     workflowid: int | None = Field(default=None, foreign_key='dbworkflow.id', primary_key=True)

class DBPoll(DBBase, ppp.PollSpec, sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = list_default()
    attrs: dict[str, Union[str, int, float]] = dict_default()
    nchain: int = -1
    files: list['DBFile'] = sqlmodel.Relationship(back_populates='poll')
    reviews: list['DBReview'] = sqlmodel.Relationship(back_populates='poll')
    workflowid: int = sqlmodel.Field(foreign_key='dbworkflow.id')
    workflow: 'DBWorkFlow' = sqlmodel.Relationship(back_populates='polls')

    def validated_with_backend(self, backend):
        if conflicts := set(backend.select(DBPoll, name=self.name, _idnot=self.id)):
            print('conflicts', [c.name for c in conflicts])
            raise DuplicateError(f'duplicate poll {self.name}', conflicts)
        return self

    def clear(self, backend):
        check_ghost_poll_and_file(backend)
        for r in backend.select(DBReview, pollid=self.id):
            r.pollid = 666
            r.fileid = 666
        backend.session.commit()
        for f in backend.select(DBFile, pollid=self.id):
            backend.session.delete(f)

class DBFile(DBBase, ppp.FileSpec, sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = list_default()
    attrs: dict[str, Union[str, int, float]] = dict_default()
    pollid: int = sqlmodel.Field(default=None, foreign_key='dbpoll.id')
    poll: DBPoll = sqlmodel.Relationship(back_populates='files')
    reviews: list['DBReview'] = sqlmodel.Relationship(back_populates='file')

    def validated_with_backend(self, backend):
        # assert os.path.exists(self.fname)
        return self

    @pydantic.validator('fname')
    def valfname(cls, fname):
        return os.path.abspath(fname)

class DBReview(DBBase, ppp.ReviewSpec, sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = list_default()
    attrs: dict[str, Union[str, int, float]] = dict_default()
    fileid: int = sqlmodel.Field(default=None, foreign_key='dbfile.id')
    pollid: int = sqlmodel.Field(default=None, foreign_key='dbpoll.id')
    workflowbbkey: int = sqlmodel.Field(default=None, foreign_key='dbworkflow.id')
    file: DBFile = sqlmodel.Relationship(back_populates='reviews')
    poll: DBPoll = sqlmodel.Relationship(back_populates='reviews')
    workflow: 'DBWorkFlow' = sqlmodel.Relationship(back_populates='reviews')
    steps: list['DBReviewStep'] = sqlmodel.Relationship(back_populates='review')

    def validated_with_backend(self, backend):
        assert self.file
        assert self.poll

class DBReviewStep(DBBase, ppp.ReviewStepSpec, sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = list_default()
    attrs: dict[str, Union[str, int, float]] = dict_default()
    task: dict[str, Union[str, int, float]] = dict_default()
    reviewid: int = sqlmodel.Field(default=None, foreign_key='dbreview.id')
    review: DBReview = sqlmodel.Relationship(back_populates='steps')
    flowstepid: int = sqlmodel.Field(default=None, foreign_key='dbflowstep.id')
    flowstep: 'DBFlowStep' = sqlmodel.Relationship(back_populates='reviews')

class DBPymolCMD(DBBase, ppp.PymolCMDSpec, sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = list_default()
    attrs: dict[str, Union[str, int, float]] = dict_default()
    flowsteps: list['DBFlowStep'] = sqlmodel.Relationship(back_populates='cmds',
                                                          link_model=DBPymolCMDFlowStepLink)

    def validated_with_backend(self, backend):
        if conflicts := set(backend.select(DBPymolCMD, name=self.name, _idnot=self.id)):
            raise DuplicateError(f'duplicate pymolcmd {self.name}', conflicts)
        return self

class DBFlowStep(DBBase, ppp.FlowStepSpec, sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = list_default()
    attrs: dict[str, Union[str, int, float]] = dict_default()
    taskgen: dict[str, Union[str, int, float]] = dict_default()
    workflowid: int = sqlmodel.Field(foreign_key='dbworkflow.id')
    workflow: 'DBWorkFlow' = sqlmodel.Relationship(back_populates='flowsteps')
    cmds: list['DBPymolCMD'] = sqlmodel.Relationship(back_populates='flowsteps',
                                                     link_model=DBPymolCMDFlowStepLink)
    reviews: list['DBReviewStep'] = sqlmodel.Relationship(back_populates='flowstep00')

class DBWorkflow(DBBase, ppp.WorkflowSpec, sqlmodel.SQLModel, table=True):
    id: Optional[int] = sqlmodel.Field(default=None, primary_key=True)
    props: list[str] = list_default()
    attrs: dict[str, Union[str, int, float]] = dict_default()
    flowsteps: list['DBFlowStep'] = sqlmodel.Relationship(back_populates='workflow')
    polls: list['DBPoll'] = sqlmodel.Relationship(back_populates='workflow')
    reviews: list['DBReview'] = sqlmodel.Relationship(back_populates='workflow')

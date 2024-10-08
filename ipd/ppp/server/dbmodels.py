import sys
import os
from datetime import datetime
from typing import Optional
import ipd
from ipd import ppp
from typing import Union

pydantic = ipd.lazyimport('pydantic', pip=True)
sqlmodel = ipd.lazyimport('sqlmodel', pip=True)._import_module()
sqlalchemy = ipd.lazyimport('sqlalchemy', pip=True)._import_module()
from sqlmodel import Relationship, SQLModel, Field

class DBBase(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)

    def __hash__(self):
        return self.id

    def clear(self, backend):
        return

    def validated_with_backend(self, backend):
        return self

Props = list[str]
Attrs = dict[str, Union[str, int, float]]
list_default = lambda: Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=list)
dict_default = lambda: Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=dict)

class DuplicateError(Exception):
    def __init__(self, msg, conflict):
        super().__init__(msg)
        self.conflict = conflict

class DBPoll(DBBase, ppp.PollSpec, table=True):
    props: Props = list_default()
    attrs: Attrs = dict_default()
    nchain: int = -1
    files: list['DBFile'] = Relationship(back_populates='poll')
    reviews: list['DBReview'] = Relationship(back_populates='poll')
    workflowid: int = Field(foreign_key='dbworkflow.id')
    workflow: 'DBWorkflow' = Relationship(back_populates='polls')

    def validated_with_backend(self, backend):
        if conflicts := set(backend.select(DBPoll, name=self.name, idnot=self.id)):
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

class DBFile(DBBase, ppp.FileSpec, table=True):
    props: Props = list_default()
    attrs: Attrs = dict_default()
    pollid: int = Field(default=None, foreign_key='dbpoll.id')
    poll: DBPoll = Relationship(back_populates='files')
    reviews: list['DBReview'] = Relationship(back_populates='file')

    @pydantic.validator('fname')
    def valfname(cls, fname):
        return os.path.abspath(fname)

class DBReview(DBBase, ppp.ReviewSpec, table=True):
    props: Props = list_default()
    attrs: Attrs = dict_default()
    fileid: int = Field(default=None, foreign_key='dbfile.id')
    pollid: int = Field(default=None, foreign_key='dbpoll.id')
    workflowid: int = Field(default=None, foreign_key='dbworkflow.id')
    file: DBFile = Relationship(back_populates='reviews')
    poll: DBPoll = Relationship(back_populates='reviews')
    workflow: 'DBWorkflow' = Relationship(back_populates='reviews')
    steps: list['DBReviewStep'] = Relationship(back_populates='review')

class DBReviewStep(DBBase, ppp.ReviewStepSpec, table=True):
    props: Props = list_default()
    attrs: Attrs = dict_default()
    task: Attrs = dict_default()
    reviewid: int = Field(default=None, foreign_key='dbreview.id')
    review: DBReview = Relationship(back_populates='steps')
    flowstepid: int = Field(default=None, foreign_key='dbflowstep.id')
    flowstep: 'DBFlowStep' = Relationship(back_populates='reviews')

class DBPymolCMDFlowStepLink(SQLModel, table=True):
    pymolcmdid: int | None = Field(default=None, foreign_key='dbpymolcmd.id', primary_key=True)
    flowstepid: int | None = Field(default=None, foreign_key='dbflowstep.id', primary_key=True)

class DBPymolCMD(DBBase, ppp.PymolCMDSpec, table=True):
    props: Props = list_default()
    attrs: Attrs = dict_default()
    flowsteps: list['DBFlowStep'] = Relationship(back_populates='cmds', link_model=DBPymolCMDFlowStepLink)

    def validated_with_backend(self, backend):
        if conflicts := set(backend.select(DBPymolCMD, name=self.name, idnot=self.id)):
            raise DuplicateError(f'duplicate pymolcmd {self.name}', conflicts)
        return self

class DBFlowStep(DBBase, ppp.FlowStepSpec, table=True):
    props: Props = list_default()
    attrs: Attrs = dict_default()
    taskgen: Attrs = dict_default()
    workflowid: int = Field(foreign_key='dbworkflow.id')
    workflow: 'DBWorkflow' = Relationship(back_populates='flowsteps')
    cmds: list['DBPymolCMD'] = Relationship(back_populates='flowsteps', link_model=DBPymolCMDFlowStepLink)
    reviews: list['DBReviewStep'] = Relationship(back_populates='flowstep')

class DBWorkflow(DBBase, ppp.WorkflowSpec, table=True):
    props: Props = list_default()
    attrs: Attrs = dict_default()
    flowsteps: list['DBFlowStep'] = Relationship(back_populates='workflow')
    polls: list['DBPoll'] = Relationship(back_populates='workflow')
    reviews: list['DBReview'] = Relationship(back_populates='workflow')

class DBUserUserLink(SQLModel, table=True):
    followerid: int | None = Field(default=None, foreign_key='dbuser.id', primary_key=True)
    followingid: int | None = Field(default=None, foreign_key='dbuser.id', primary_key=True)

class DBUserGroupLink(SQLModel, table=True):
    userid: int | None = Field(default=None, foreign_key='dbuser.id', primary_key=True)
    groupid: int | None = Field(default=None, foreign_key='dbgroup.id', primary_key=True)

class DBUser(DBBase, ppp.UserSpec, table=True):
    props: Props = list_default()
    attrs: Attrs = dict_default()
    groups: list['DBGroup'] = Relationship(back_populates='users', link_model=DBUserGroupLink)
    followers: list['DBUser'] = Relationship(
        back_populates='following',
        link_model=DBUserUserLink,
        sa_relationship_kwargs=dict(
            primaryjoin="DBUser.id==DBUserUserLink.followerid",
            secondaryjoin="DBUser.id==DBUserUserLink.followingid",
        ),
    )
    following: list['DBUser'] = Relationship(
        back_populates='followers',
        link_model=DBUserUserLink,
        sa_relationship_kwargs=dict(
            primaryjoin="DBUser.id==DBUserUserLink.followingid",
            secondaryjoin="DBUser.id==DBUserUserLink.followerid",
        ),
    )

class DBGroup(DBBase, ppp.UserSpec, table=True):
    props: Props = list_default()
    attrs: Attrs = dict_default()
    users: list['DBUser'] = Relationship(back_populates='groups', link_model=DBUserGroupLink)

assert 0

from ipd.ppp.models import *

def DBPoll_clear(self, backend, ghost=True):
    for r in backend.select(DBReview, pollid=self.id):
        if not ghost: r.pollid, r.pollfileid = 1, 1
    backend.session.commit()
    for f in backend.select(DBPollFile, pollid=self.id):
        if ghost: f.ghost = True
        else: backend.session.delete(f)

backend_models['poll'].clear = DBPoll_clear

import uuid
from typing import Optional

import ipd
from ipd import ppp
from ipd.crud.backend import Attrs, BackendModelBase, Props, attrs_default, props_default

backend_models = ipd.crud.backend.make_backend_models(ipd.ppp.spec_models)
for cls in backend_models.values():
    globals()[cls.__name__] = cls

pydantic = ipd.lazyimport('pydantic', pip=True)
sqlmodel = ipd.lazyimport('sqlmodel', pip=True)._import_module()
sqlalchemy = ipd.lazyimport('sqlalchemy', pip=True)._import_module()
from sqlmodel import Field, Relationship, SQLModel

class _DBWithUser(BackendModelBase):
    userid: uuid.UUID = Field(foreign_key='dbuser.id')

class DuplicateError(Exception):
    def __init__(self, msg, conflict):
        super().__init__(msg)
        self.conflict = conflict

class DBPoll(_DBWithUser, ppp.PollSpec, table=True):
    props: Props = props_default()
    attrs: Attrs = attrs_default()
    name: str = Field(sa_column_kwargs={"unique": True})
    pollfiles: list['DBPollFile'] = Relationship(back_populates='poll')
    reviews: list['DBReview'] = Relationship(back_populates='poll')
    workflowid: Optional[uuid.UUID] = Field(foreign_key='dbworkflow.id', nullable=True)
    workflow: Optional['DBWorkflow'] = Relationship(back_populates='polls')
    user: 'DBUser' = Relationship(back_populates='polls')

class DBFileKind(BackendModelBase, ppp.FileKindSpec, table=True):
    props: Props = props_default()
    attrs: Attrs = attrs_default()
    pollfiles: list['DBPollFile'] = Relationship(back_populates='filekind')

class DBPollFile(BackendModelBase, ppp.PollFileSpec, table=True):
    props: Props = props_default()
    attrs: Attrs = attrs_default()
    pollid: uuid.UUID = Field(default=None, foreign_key='dbpoll.id')
    poll: DBPoll = Relationship(back_populates='pollfiles')
    filekindid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbfilekind.id', nullable=True)
    filekind: Optional[DBFileKind] = Relationship(back_populates='pollfiles')
    reviews: list['DBReview'] = Relationship(back_populates='pollfile')
    children: list['DBPollFile'] = Relationship(back_populates='parent')
    parentid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbpollfile.id', nullable=True)
    parent: Optional['DBPollFile'] = Relationship(back_populates='children',
                                                  sa_relationship_kwargs=dict(cascade="all",
                                                                              remote_side='DBPollFile.id'))

class DBReview(_DBWithUser, ppp.ReviewSpec, table=True):
    props: Props = props_default()
    attrs: Attrs = attrs_default()
    pollfileid: uuid.UUID = Field(foreign_key='dbpollfile.id')
    pollfile: DBPollFile = Relationship(back_populates='reviews')
    pollid: uuid.UUID = Field(default=None, foreign_key='dbpoll.id')
    poll: DBPoll = Relationship(back_populates='reviews')
    workflowid: uuid.UUID = Field(foreign_key='dbworkflow.id', nullable=True)
    workflow: 'DBWorkflow' = Relationship(back_populates='reviews')
    steps: list['DBReviewStep'] = Relationship(back_populates='review')
    user: 'DBUser' = Relationship(back_populates='reviews')

class DBReviewStep(BackendModelBase, ppp.ReviewStepSpec, table=True):
    props: Props = props_default()
    attrs: Attrs = attrs_default()
    task: Attrs = attrs_default()
    reviewid: uuid.UUID = Field(default=None, foreign_key='dbreview.id')
    review: DBReview = Relationship(back_populates='steps')
    flowstepid: uuid.UUID = Field(default=None, foreign_key='dbflowstep.id')
    flowstep: 'DBFlowStep' = Relationship(back_populates='reviews')

class DBPymolCMDFlowStepLink(SQLModel, table=True):
    pymolcmdid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbpymolcmd.id', primary_key=True)
    flowstepid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbflowstep.id', primary_key=True)

class DBPymolCMD(_DBWithUser, ppp.PymolCMDSpec, table=True):
    name: str = Field(sa_column_kwargs={"unique": True})
    props: Props = props_default()
    attrs: Attrs = attrs_default()
    flowsteps: list['DBFlowStep'] = Relationship(back_populates='cmds', link_model=DBPymolCMDFlowStepLink)
    user: 'DBUser' = Relationship(back_populates='cmds')

class DBFlowStep(BackendModelBase, ppp.FlowStepSpec, table=True):
    props: Props = props_default()
    attrs: Attrs = attrs_default()
    taskgen: Attrs = attrs_default()
    workflowid: uuid.UUID = Field(foreign_key='dbworkflow.id')
    workflow: 'DBWorkflow' = Relationship(back_populates='flowsteps')
    cmds: list['DBPymolCMD'] = Relationship(back_populates='flowsteps', link_model=DBPymolCMDFlowStepLink)
    reviews: list['DBReviewStep'] = Relationship(back_populates='flowstep')

class DBWorkflow(_DBWithUser, ppp.WorkflowSpec, table=True):
    name: str = Field(sa_column_kwargs={"unique": True})
    props: Props = props_default()
    attrs: Attrs = attrs_default()
    flowsteps: list['DBFlowStep'] = Relationship(back_populates='workflow')
    polls: list['DBPoll'] = Relationship(back_populates='workflow')
    reviews: list['DBReview'] = Relationship(back_populates='workflow')
    user: 'DBUser' = Relationship(back_populates='workflows')

class DBUserUserLink(SQLModel, table=True):
    followerid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbuser.id', primary_key=True)
    followingid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbuser.id', primary_key=True)

class DBUserGroupLink(SQLModel, table=True):
    userid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbuser.id', primary_key=True)
    groupid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbgroup.id', primary_key=True)

class DBUser(BackendModelBase, ppp.UserSpec, table=True):
    name: str = Field(sa_column_kwargs={"unique": True})
    props: Props = props_default()
    attrs: Attrs = attrs_default()
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
    polls: list['DBPoll'] = Relationship(back_populates='user')
    reviews: list['DBReview'] = Relationship(back_populates='user')
    cmds: list['DBPymolCMD'] = Relationship(back_populates='user')
    workflows: list['DBWorkflow'] = Relationship(back_populates='user')
    ownedgroups: list['DBGroup'] = Relationship(back_populates='user')

class DBGroup(_DBWithUser, ppp.GroupSpec, table=True):
    name: str = Field(sa_column_kwargs={"unique": True})
    props: Props = props_default()
    attrs: Attrs = attrs_default()
    users: list['DBUser'] = Relationship(back_populates='groups', link_model=DBUserGroupLink)
    user: 'DBUser' = Relationship(back_populates='ownedgroups')

backend_models = {name: globals()[f'DB{spec.__name__[:-4]}'] for name, spec in ipd.ppp.spec_models.items()}
client_models = ipd.crud.frontend.make_client_models(ipd.ppp.spec_models, backend_models)

def DBPoll_clear(self, backend, ghost=True):
    for r in backend.select(DBReview, pollid=self.id):
        if not ghost: r.pollid, r.pollfileid = 1, 1
    backend.session.commit()
    for f in backend.select(DBPollFile, pollid=self.id):
        if ghost: f.ghost = True
        else: backend.session.delete(f)

backend_models['poll'].clear = DBPoll_clear

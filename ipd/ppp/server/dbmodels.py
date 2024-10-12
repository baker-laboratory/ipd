import os
from typing import Optional
import ipd
from ipd import ppp
from typing import Union, Optional

pydantic = ipd.lazyimport('pydantic', pip=True)
sqlmodel = ipd.lazyimport('sqlmodel', pip=True)._import_module()
sqlalchemy = ipd.lazyimport('sqlalchemy', pip=True)._import_module()
from sqlmodel import Relationship, SQLModel, Field

_Props = list[str]
_Attrs = dict[str, Union[str, int, float]]
_list_default = lambda: Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=list)
_dict_default = lambda: Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=dict)

class _DBBase(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)

    def __hash__(self):
        return self.id

    def clear(self, backend, ghost=True):
        return

    def validated_with_backend(self, backend):
        return self

class _DBWithUser(_DBBase):
    userid: int = Field(foreign_key='dbuser.id')

    @pydantic.field_validator('userid')
    def valuserid(userid):
        if isinstance(userid, str):
            userid = backend.user(name=userid)
        return userid

class DuplicateError(Exception):
    def __init__(self, msg, conflict):
        super().__init__(msg)
        self.conflict = conflict

class DBPoll(_DBWithUser, ppp.PollSpec, table=True):
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
    nchain: int = -1
    name: str = Field(sa_column_kwargs={"unique": True})
    pollfiles: list['DBPollFile'] = Relationship(back_populates='poll')
    reviews: list['DBReview'] = Relationship(back_populates='poll')
    workflowid: int = Field(foreign_key='dbworkflow.id', default=1)
    workflow: 'DBWorkflow' = Relationship(back_populates='polls')
    user: 'DBUser' = Relationship(back_populates='polls')

    def clear(self, backend, ghost=True):
        for r in backend.select(DBReview, pollid=self.id):
            if not ghost: r.pollid, r.pollfileid = 1, 1
        backend.session.commit()
        for f in backend.select(DBPollFile, pollid=self.id):
            if ghost: f.ghost = True
            else: backend.session.delete(f)

class DBFileKind(_DBWithUser, ppp.FileKindSpec, table=True):
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
    pollfiles: list['DBPollFile'] = Relationship(back_populates='filekind')

class DBPollFile(_DBBase, ppp.PollFileSpec, table=True):
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
    pollid: int = Field(default=None, foreign_key='dbpoll.id')
    poll: DBPoll = Relationship(back_populates='pollfiles')
    filekindid: Optional[int] = Field(default=None, foreign_key='dbfilekind.id', nullable=True)
    filekind: DBFileKind = Relationship(back_populates='pollfiles')
    reviews: list['DBReview'] = Relationship(back_populates='pollfile')
    parentid: Optional[int] = Field(default=None, foreign_key='dbpollfile.id', nullable=True)
    parent: Optional['DBPollFile'] = Relationship(back_populates='children',
                                                  sa_relationship_kwargs=dict(cascade="all",
                                                                              remote_side='DBPollFile.id'))
    children: list['DBPollFile'] = Relationship(back_populates='parent')

    @pydantic.validator('fname')
    def valfname(cls, fname):
        return os.path.abspath(fname)

class DBReview(_DBWithUser, ppp.ReviewSpec, table=True):
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
    pollfileid: int = Field(default=None, foreign_key='dbpollfile.id')
    pollfile: DBPollFile = Relationship(back_populates='reviews')
    pollid: int = Field(default=None, foreign_key='dbpoll.id')
    poll: DBPoll = Relationship(back_populates='reviews')
    workflowid: int = Field(default=None, foreign_key='dbworkflow.id')
    workflow: 'DBWorkflow' = Relationship(back_populates='reviews')
    steps: list['DBReviewStep'] = Relationship(back_populates='review')
    user: 'DBUser' = Relationship(back_populates='reviews')

class DBReviewStep(_DBBase, ppp.ReviewStepSpec, table=True):
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
    task: _Attrs = _dict_default()
    reviewid: int = Field(default=None, foreign_key='dbreview.id')
    review: DBReview = Relationship(back_populates='steps')
    flowstepid: int = Field(default=None, foreign_key='dbflowstep.id')
    flowstep: 'DBFlowStep' = Relationship(back_populates='reviews')

class DBPymolCMDFlowStepLink(SQLModel, table=True):
    pymolcmdid: int | None = Field(default=None, foreign_key='dbpymolcmd.id', primary_key=True)
    flowstepid: int | None = Field(default=None, foreign_key='dbflowstep.id', primary_key=True)

class DBPymolCMD(_DBWithUser, ppp.PymolCMDSpec, table=True):
    name: str = Field(sa_column_kwargs={"unique": True})
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
    flowsteps: list['DBFlowStep'] = Relationship(back_populates='cmds', link_model=DBPymolCMDFlowStepLink)
    user: 'DBUser' = Relationship(back_populates='cmds')

class DBFlowStep(_DBBase, ppp.FlowStepSpec, table=True):
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
    taskgen: _Attrs = _dict_default()
    workflowid: int = Field(foreign_key='dbworkflow.id')
    workflow: 'DBWorkflow' = Relationship(back_populates='flowsteps')
    cmds: list['DBPymolCMD'] = Relationship(back_populates='flowsteps', link_model=DBPymolCMDFlowStepLink)
    reviews: list['DBReviewStep'] = Relationship(back_populates='flowstep')

class DBWorkflow(_DBWithUser, ppp.WorkflowSpec, table=True):
    name: str = Field(sa_column_kwargs={"unique": True})
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
    flowsteps: list['DBFlowStep'] = Relationship(back_populates='workflow')
    polls: list['DBPoll'] = Relationship(back_populates='workflow')
    reviews: list['DBReview'] = Relationship(back_populates='workflow')
    user: 'DBUser' = Relationship(back_populates='workflows')

class DBUserUserLink(SQLModel, table=True):
    followerid: int | None = Field(default=None, foreign_key='dbuser.id', primary_key=True)
    followingid: int | None = Field(default=None, foreign_key='dbuser.id', primary_key=True)

class DBUserGroupLink(SQLModel, table=True):
    userid: int | None = Field(default=None, foreign_key='dbuser.id', primary_key=True)
    groupid: int | None = Field(default=None, foreign_key='dbgroup.id', primary_key=True)

class DBUser(_DBBase, ppp.UserSpec, table=True):
    name: str = Field(sa_column_kwargs={"unique": True})
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
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
    props: _Props = _list_default()
    attrs: _Attrs = _dict_default()
    users: list['DBUser'] = Relationship(back_populates='groups', link_model=DBUserGroupLink)
    user: 'DBUser' = Relationship(back_populates='ownedgroups')

backend_model = {name: globals()[f'DB{spec.__name__[:-4]}'] for name, spec in ipd.ppp.spec_model.items()}

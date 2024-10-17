import ipd
import pydantic
import sqlmodel
import sqlalchemy
from typing import Optional
import uuid
import rich

def main():
    # test_many2many_sanity_check()
    test_many2many_basic()
    print('test_crud PASS')

def test_many2many_basic():
    class LocalSQLModel(sqlmodel.SQLModel):
        metadata = sqlmodel.MetaData()

    class UserSpec(ipd.crud.SpecBase):
        id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
        groups: list['GroupSpec']

    class GroupSpec(ipd.crud.SpecBase):
        id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
        users: list['UserSpec']

    spec_models = dict(user=UserSpec, group=GroupSpec)
    assert not any(name.endswith('s') for name in spec_models)
    backend_models, links, remoteprops = ipd.crud.backend.make_backend_models(spec_models, LocalSQLModel)
    client_models = ipd.crud.frontend.make_client_models(spec_models, backend_models, remoteprops)
    print(backend_models['user'].__mro__)
    helper_test_users_groups(LocalSQLModel, **{c.__name__: c for c in backend_models.values()}, **links)

def test_many2many_sanity_check():
    class LocalSQLModel(sqlmodel.SQLModel):
        metadata = sqlmodel.MetaData()

    linkbody = dict(userid=sqlmodel.Field(default=None, foreign_key='dbuser.id', primary_key=True),
                    groupid=sqlmodel.Field(default=None, foreign_key='dbgroup.id', primary_key=True),
                    __annotations__=dict(userid=Optional[uuid.UUID], groupid=Optional[uuid.UUID]))
    Link = type('Link', (LocalSQLModel, ), linkbody, table=True)
    userbody = dict(id=sqlmodel.Field(primary_key=True, default_factory=uuid.uuid4),
                    groups=sqlmodel.Relationship(back_populates='users', link_model=Link),
                    __annotations__=dict(id=uuid.UUID, groups=list['DBGroup']))
    DBUser = type('DBUser', (LocalSQLModel, ), userbody, table=True)
    groupbody = dict(id=sqlmodel.Field(primary_key=True, default_factory=uuid.uuid4),
                     users=sqlmodel.Relationship(back_populates='groups', link_model=Link),
                     __annotations__=dict(id=uuid.UUID, users=list['DBUser']))
    DBGroup = type('DBGroup', (LocalSQLModel, ), groupbody, table=True)
    print('Link')
    rich.print(linkbody)
    print('User')
    rich.print(userbody)
    print('Group')
    rich.print(groupbody)
    helper_test_users_groups(LocalSQLModel, DBUser, DBGroup, GroupUserDefaultLink=Link)

def helper_test_users_groups(LocalSQLModel, DBUser, DBGroup, GroupUserDefaultLink, **kw):
    # print(DBGroup.__table__.columns)
    engine = sqlmodel.create_engine('sqlite:////tmp/test.db')
    for k, v in LocalSQLModel.metadata.tables.items():
        print(k, type(v))
    LocalSQLModel.metadata.create_all(engine)
    session = sqlmodel.Session(engine)

    users = [DBUser() for _ in range(10)]
    groups = [DBGroup() for _ in range(10)]
    users[0].groups.append(groups[2])
    for u in users:
        session.add(u)
    for g in groups:
        session.add(g)
    session.commit()

if __name__ == '__main__':
    main()

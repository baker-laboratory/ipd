import tempfile
from typing import Optional, Type
from uuid import UUID, uuid4

import pydantic
import pytest
import sqlmodel.pool
from fastapi.testclient import TestClient
from sqlalchemy.orm import registry

import ipd

def main():
    for k, v in globals().copy().items():
        if not k.startswith('test_'): continue
        with tempfile.TemporaryDirectory() as td:
            v(td)
    print('test_crud PASS')

def create_new_sqlmodel_base() -> Type[sqlmodel.SQLModel]:
    # mapper_registry = sqlalchemy.orm.registry()
    # Base = mapper_registry.generate_base()
    # Base.registry._class_registry.clear()

    # __abstract__ = True  # This marks it as an abstract class, so it won't create its own table
    # metadata = Base.metadata
    # _sa_registry = copy.deepcopy(sqlmodel.SQLModel._sa_registry)
    NewBase = type('NewBase', (sqlmodel.SQLModel, ), {}, registry=registry())
    # NewBase._sa_registry._class_registry.clear()

    return NewBase

@pytest.mark.fast
def test_user_group(tmpdir):
    LocalSQLModel = create_new_sqlmodel_base()

    class _SpecWithUser(ipd.crud.SpecBase):
        userid: ipd.crud.ModelRef['UserZSpec'] = pydantic.Field(default='anonymous_coward',
                                                                validate_default=True)
        ispublic: bool = True
        telemetry: bool = False

    class PollZSpec(_SpecWithUser):
        pass

    class UserZSpec(ipd.crud.SpecBase):
        name: ipd.crud.Unique[str]
        fullname: str = ''
        number: int = 0
        someid: UUID = pydantic.Field(default_factory=uuid4)
        followers: list['UserZSpec'] = []
        following: list['UserZSpec'] = []
        groups: list['GroupZSpec'] = []

    class GroupZSpec(_SpecWithUser):
        name: ipd.crud.Unique[str]
        users: list['UserZSpec'] = []
        userid: ipd.crud.ModelRef['UserZSpec', 'ownedgroups'] = pydantic.Field(default='anonymous_coward',
                                                                               validate_default=True)

    models = dict(pollz=PollZSpec, userz=UserZSpec, groupz=GroupZSpec)
    MyBackend = type('MyBackend', (ipd.crud.BackendBase, ), {}, models=models)
    MyClient = type('MyClient', (ipd.crud.ClientBase, ), {}, backend=MyBackend)

    backend = MyBackend(f'sqlite:///{tmpdir}/test.db')
    print('backend.newuserz', backend.newuserz(name='foo'))
    print('backend.newuserz', backend.newuserz(name='bar'))
    print('backend.newuserz', backend.newuserz(name='baz'))
    assert 3 == len(backend.userzs())
    testclient = TestClient(backend.app)
    assert testclient.get('/api/userzs').status_code == 200
    client = MyClient(testclient)

    client.upload(UserZSpec(name='boo'))
    a, b, c, d = client.userzs()
    assert a.name == 'foo'
    assert b.fullname == ''
    b.fullname = 'foo bar baz'
    assert b.fullname == 'foo bar baz'
    c.number = 7
    assert c.number == 7
    oldid, newid = d.someid, uuid4()
    d.someid = newid
    assert d.someid == newid
    assert newid != oldid
    a.followers.extend([b, c, d])
    assert a in b.following
    assert a in c.following
    assert a in d.following

@pytest.mark.fast
def test_many2many_basic(tmpdir):
    LocalSQLModel = create_new_sqlmodel_base()

    class UserCSpec(ipd.crud.SpecBase):
        id: UUID = pydantic.Field(default_factory=uuid4)
        groups: list['GroupCSpec'] = []

    class GroupCSpec(ipd.crud.SpecBase):
        id: UUID = pydantic.Field(default_factory=uuid4)
        users: list['UserCSpec'] = []

    models = dict(userc=UserCSpec, groupc=GroupCSpec)
    MyBackend = type('MyBackend', (ipd.crud.BackendBase, ), {}, models=models, SQL=LocalSQLModel)
    helper_test_users_groups(tmpdir, LocalSQLModel, MyBackend.DBUserC, MyBackend.DBGroupC)

@pytest.mark.fast
def test_one2many_parent(tmpdir):
    LocalSQLModel = create_new_sqlmodel_base()

    class ParentChildSpec(ipd.crud.SpecBase):
        parentid: ipd.crud.ModelRef['ParentChildSpec', 'children'] = None

    models = dict(parentchild=ParentChildSpec)
    MyBackend = type('MyBackend', (ipd.crud.BackendBase, ), {}, models=models, SQL=LocalSQLModel)
    session = helper_create_db(tmpdir, LocalSQLModel)
    a = MyBackend.DBParentChild()
    b = MyBackend.DBParentChild()
    c = MyBackend.DBParentChild()
    session.add(a)
    session.add(b)
    session.add(c)
    b.parent = a
    b.children.append(c)
    assert b.parent.id == a.id
    assert a.children[0].id == b.id
    assert c.parent.id == b.id
    assert c.id == b.children[0].id
    session.commit()
    assert b.parent.id == a.id
    assert a.children[0].id == b.id
    assert c.parent.id == b.id
    assert c.id == b.children[0].id

@pytest.mark.fast
def test_many2many_parent(tmpdir):
    LocalSQLModel = create_new_sqlmodel_base()

    class UserBSpec(ipd.crud.SpecBase):
        followers: list['UserBSpec'] = []
        following: list['UserBSpec'] = []

    models = dict(userb=UserBSpec)
    MyBackend = type('MyBackend', (ipd.crud.BackendBase, ), {}, models=models, SQL=LocalSQLModel)
    session = helper_create_db(tmpdir, LocalSQLModel)
    a = MyBackend.DBUserB(name='a')
    b = MyBackend.DBUserB(name='b')
    c = MyBackend.DBUserB(name='c')
    d = MyBackend.DBUserB(name='b')
    session.add(a)
    session.add(b)
    session.add(d)
    b.following.append(a)
    b.followers.append(c)
    b.followers.append(d)
    assert a.id in {_.id for _ in b.following}
    assert b.id in {_.id for _ in a.followers}
    assert b.id in {_.id for _ in c.following}
    assert b.id in {_.id for _ in d.following}
    assert c.id in {_.id for _ in b.followers}
    assert d.id in {_.id for _ in b.followers}
    session.commit()
    assert a.id in {_.id for _ in b.following}
    assert b.id in {_.id for _ in a.followers}
    assert b.id in {_.id for _ in c.following}
    assert b.id in {_.id for _ in d.following}
    assert c.id in {_.id for _ in b.followers}
    assert d.id in {_.id for _ in b.followers}

@pytest.mark.fast
def test_many2many_sanity_check(tmpdir):
    LocalSQLModel = create_new_sqlmodel_base()

    linkbody = dict(useraid=sqlmodel.Field(default=None, foreign_key='dbusera.id', primary_key=True),
                    groupaid=sqlmodel.Field(default=None, foreign_key='dbgroupa.id', primary_key=True),
                    __annotations__=dict(useraid=Optional[UUID], groupaid=Optional[UUID]))
    Link = type('LinkA', (LocalSQLModel, ), linkbody, table=True)
    userbody = dict(id=sqlmodel.Field(primary_key=True, default_factory=uuid4),
                    groups=sqlmodel.Relationship(back_populates='users', link_model=Link),
                    __annotations__=dict(id=UUID, groups=list['DBGroupA']))
    DBUserA = type('DBUserA', (LocalSQLModel, ), userbody, table=True)
    groupbody = dict(id=sqlmodel.Field(primary_key=True, default_factory=uuid4),
                     users=sqlmodel.Relationship(back_populates='groups', link_model=Link),
                     __annotations__=dict(id=UUID, users=list['DBUserA']))
    DBGroupA = type('DBGroupA', (LocalSQLModel, ), groupbody, table=True)
    # print('Link')
    # rich.print(linkbody)
    # print('User')
    # rich.print(userbody)
    # print('Group')
    # rich.print(groupbody)
    helper_test_users_groups(tmpdir, LocalSQLModel, DBUserA, DBGroupA)

def helper_create_db(tmpdir, LocalSQLModel):
    engine = sqlmodel.create_engine(
        f'sqlite:///{tmpdir}/test.db',
        # connect_args={"check_same_thread": False},
        # poolclass=sqlmodel.pool.StaticPool,
    )
    print('metadata id', id(LocalSQLModel.metadata))
    LocalSQLModel.metadata.create_all(engine)
    session = sqlmodel.Session(engine)
    return (session)

def helper_test_users_groups(tmpdir, LocalSQLModel, dbusertype, dbgrouptype):
    # print(dbgrouptype.__table__.columns)
    session = helper_create_db(tmpdir, LocalSQLModel)
    users = [dbusertype() for _ in range(10)]
    groups = [dbgrouptype() for _ in range(10)]
    users[0].groups.append(groups[2])
    for u in users:
        session.add(u)
    for g in groups:
        session.add(g)
    session.commit()
    assert users[0].id == groups[2].users[0].id

if __name__ == '__main__':
    main()

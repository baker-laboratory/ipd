import copy
from fastapi.testclient import TestClient
import ipd
import pydantic
import sqlmodel
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from typing import Optional, Type
import uuid
import rich
import tempfile

def main():
    for fn in (
            # test_many2many_sanity_check,
            # test_many2many_basic,
            # test_one2many_parent,
            # test_many2many_parent,
            test_user_group, ):
        with tempfile.TemporaryDirectory() as tempdir:
            fn(tempdir)
    print('test_crud PASS')

def create_new_sqlmodel_base() -> Type[sqlmodel.SQLModel]:
    mapper_registry = sqlalchemy.orm.registry()
    Base = mapper_registry.generate_base()
    Base.registry._class_registry.clear()

    class NewBase(sqlmodel.SQLModel):
        __abstract__ = True  # This marks it as an abstract class, so it won't create its own table
        metadata = Base.metadata
        _sa_registry = copy.deepcopy(sqlmodel.SQLModel._sa_registry)

    NewBase._sa_registry._class_registry.clear()

    return NewBase

def test_user_group(tempdir):
    LocalSQLModel = create_new_sqlmodel_base()

    class _SpecWithUser(ipd.crud.SpecBase):
        userid: ipd.crud.ModelRef['UserSpec'] = pydantic.Field(default='anonymous_coward',
                                                               validate_default=True)
        ispublic: bool = True
        telemetry: bool = False

    class UserZSpec(ipd.crud.SpecBase):
        name: ipd.crud.Unique[str]
        fullname: str = ''
        followers: list['UserZSpec'] = []
        following: list['UserZSpec'] = []
        groups: list['GroupZSpec'] = []

    class GroupZSpec(_SpecWithUser):
        name: ipd.crud.Unique[str]
        users: list['UserZSpec'] = []
        userid: ipd.crud.ModelRef['UserZSpec', 'ownedgroups'] = pydantic.Field(default='anonymous_coward',
                                                                               validate_default=True)

    models = dict(userz=UserZSpec, groupz=GroupZSpec)

    class MyBackend(ipd.crud.BackendBase, models=models):
        pass

    class MyClient(ipd.crud.ClientBase, backend=MyBackend):
        pass

    backend = MyBackend(f'{tempdir}/test.db')
    print('backend.newuserz', backend.newuserz(name='foo'))
    print('backend.newuserz', backend.newuserz(name='bar'))
    print('backend.newuserz', backend.newuserz(name='baz'))
    assert 3 == len(backend.userzs())
    testclient = TestClient(backend.app)
    assert testclient.get('/api/userzs').status_code == 200
    client = MyClient(testclient)

    client.upload(UserZSpec(name='boo'))
    a, b, c, d = client.userzs()
    a.followers.append(b)

def test_many2many_basic(tempdir):
    LocalSQLModel = create_new_sqlmodel_base()

    class UserCSpec(ipd.crud.SpecBase):
        id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
        groups: list['GroupCSpec'] = []

    class GroupCSpec(ipd.crud.SpecBase):
        id: uuid.UUID = pydantic.Field(default_factory=uuid.uuid4)
        users: list['UserCSpec'] = []

    spec_models = dict(userc=UserCSpec, groupc=GroupCSpec)
    backend_models, props, trim = ipd.crud.backend.make_backend_models(spec_models, LocalSQLModel)
    client_models = ipd.crud.frontend.make_client_models(spec_models, trim, backend_models, props)
    helper_test_users_groups(tempdir, LocalSQLModel, backend_models['userc'], backend_models['groupc'])

def test_one2many_parent(tempdir):
    LocalSQLModel = create_new_sqlmodel_base()

    class ParentChildSpec(ipd.crud.SpecBase):
        parentid: ipd.crud.ModelRef['ParentChildSpec', 'children'] = None

    spec_models = dict(parentchild=ParentChildSpec)
    backend_models, props, trim = ipd.crud.backend.make_backend_models(spec_models, LocalSQLModel)
    client_models = ipd.crud.frontend.make_client_models(spec_models, trim, backend_models, props)
    session = helper_create_db(tempdir, LocalSQLModel)
    a = backend_models['parentchild']()
    b = backend_models['parentchild']()
    c = backend_models['parentchild']()
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

def test_many2many_parent(tempdir):
    LocalSQLModel = create_new_sqlmodel_base()

    class UserBSpec(ipd.crud.SpecBase):
        followers: list['UserBSpec'] = []
        following: list['UserBSpec'] = []

    spec_models = dict(userb=UserBSpec)
    backend_models, props, trim = ipd.crud.backend.make_backend_models(spec_models, LocalSQLModel)
    client_models = ipd.crud.frontend.make_client_models(spec_models, trim, backend_models, props)
    session = helper_create_db(tempdir, LocalSQLModel)
    a = backend_models['userb']()
    b = backend_models['userb']()
    c = backend_models['userb']()
    d = backend_models['userb']()
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

def test_many2many_sanity_check(tempdir):
    LocalSQLModel = create_new_sqlmodel_base()

    linkbody = dict(useraid=sqlmodel.Field(default=None, foreign_key='dbusera.id', primary_key=True),
                    groupaid=sqlmodel.Field(default=None, foreign_key='dbgroupa.id', primary_key=True),
                    __annotations__=dict(useraid=Optional[uuid.UUID], groupaid=Optional[uuid.UUID]))
    Link = type('LinkA', (LocalSQLModel, ), linkbody, table=True)
    userbody = dict(id=sqlmodel.Field(primary_key=True, default_factory=uuid.uuid4),
                    groups=sqlmodel.Relationship(back_populates='users', link_model=Link),
                    __annotations__=dict(id=uuid.UUID, groups=list['DBGroupA']))
    DBUserA = type('DBUserA', (LocalSQLModel, ), userbody, table=True)
    groupbody = dict(id=sqlmodel.Field(primary_key=True, default_factory=uuid.uuid4),
                     users=sqlmodel.Relationship(back_populates='groups', link_model=Link),
                     __annotations__=dict(id=uuid.UUID, users=list['DBUserA']))
    DBGroupA = type('DBGroupA', (LocalSQLModel, ), groupbody, table=True)
    # print('Link')
    # rich.print(linkbody)
    # print('User')
    # rich.print(userbody)
    # print('Group')
    # rich.print(groupbody)
    helper_test_users_groups(tempdir, LocalSQLModel, DBUserA, DBGroupA)

def helper_create_db(tempdir, LocalSQLModel):
    engine = sqlmodel.create_engine(f'sqlite:///{tempdir}/test.db')
    print('metadata id', id(LocalSQLModel.metadata))
    LocalSQLModel.metadata.create_all(engine)
    session = sqlmodel.Session(engine)
    return (session)

def helper_test_users_groups(tempdir, LocalSQLModel, dbusertype, dbgrouptype):
    # print(dbgrouptype.__table__.columns)
    session = helper_create_db(tempdir, LocalSQLModel)
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

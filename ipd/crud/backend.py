from collections import defaultdict
import contextlib
import copy
from datetime import datetime
import fastapi
from icecream import ic
import inspect
import ipd
from ipd.crud.frontend import SpecBase, _ModelRefType
import operator
import os
from pathlib import Path
import pydantic
import rich
import sqlalchemy
import sqlmodel
from sqlmodel import Field
from sqlmodel.main import RelationshipInfo
import traceback
import typing
from uuid import UUID, uuid4
import yaml

# python_type_to_sqlalchemy_type = {
#     str: sqlalchemy.String,
#     int: sqlalchemy.Integer,
#     float: sqlalchemy.Float,
#     bool: sqlalchemy.Boolean,
#     # datetime.date: sqlalchemy.Date,
#     datetime: sqlalchemy.DateTime,
#     # datetime.time: sqlalchemy.Time,
#     dict: sqlalchemy.JSON,
#     list: sqlalchemy.ARRAY,
#     # decimal.Decimal: sqlalchemy.Numeric
# }

pydantic_field_args = ('default default_factory alias alias_priority validation_alias serialization_alias '
                       'title field_title_generator description examples exclude discriminator deprecated '
                       'json_schema_extra frozen validate_default repr init init_var kw_only').split()

class BackendError(Exception):
    pass

def make_backend_model_base(SQLModel=sqlmodel.SQLModel):
    class BackendModelBase(SQLModel, SpecBase):
        id: UUID = sqlmodel.Field(primary_key=True, default_factory=uuid4)

        def clear(self, backend, ghost=True):
            return

        def validated_with_backend(self, backend):
            for name, field in self.model_fields.items():
                # print(field.annotation, self[name], type(self[name]))
                if field.annotation in (UUID, typing.Optional[UUID]):
                    # print(name, self[name])
                    if name == 'id' and self[name] is None: self[name] = uuid.uuid4()
                    elif isinstance(self[name], str):
                        try:
                            self[name] = UUID(self[name])
                        except ValueError as e:
                            raise ValueError(f'bad UUID string "{self[name]}"') from e
                            # print('success')
            return self

    return BackendModelBase

class FastapiModelBackend:
    def __init_subclass__(cls, backend_models, **kw):
        super().__init_subclass__(**kw)
        cls.__backend_models__ = backend_models
        cls.__spec_models__ = {name: mdl.__spec__ for name, mdl in backend_models.items()}
        BACKEND(cls, backend_models)

    def __init__(self, engine):
        self.engine = engine
        self.session = sqlmodel.Session(self.engine)
        self.router = fastapi.APIRouter()
        self.app = fastapi.FastAPI()
        route = self.router.add_api_route
        for model in self.__backend_models__:
            route(f'/{model}', getattr(self, model), methods=['GET'])
            route(f'/{model}s', getattr(self, f'{model}s'), methods=['GET'])
            route(f'/n{model}s', getattr(self, f'n{model}s'), methods=['GET'])
            route(f'/create/{model}', getattr(self, f'create_{model}'), methods=['POST'])
        route('/getattr/{thing}/{id}/{attr}', self.getattr, methods=['GET'])
        route('/setattr/{thing}/{id}/{attr}', self.setattr, methods=['POST'])
        route('/remove/{thing}/{id}', self.remove, methods=['GET'])
        self.initdb()

        @self.app.exception_handler(Exception)
        def validation_exception_handler(request: fastapi.Request, exc: Exception):
            self.handle_error(exc)

    def _clear_all_data_for_testing_only(self):
        assert str(self.engine.url).startswith('sqlite:///')
        for mdl in self.__backend_models__.values():
            statement = sqlmodel.delete(mdl)
            result = self.session.exec(statement)
            self.session.commit()
        self.initdb()

    def initdb(self):
        sqlmodel.SQLModel.metadata.create_all(self.engine)

    def iserror(self, thing):
        return isinstance(thing, fastapi.responses.Response) and thing.status_code != 200

    def handle_error(self, exc):
        exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
        print('!' * 80)
        print(exc_str)
        print('!' * 80)
        result = f'{"!"*80}\n{exc_str}\nSTACK:\n{traceback.format_exc()}\n{"!"*80}'
        return fastapi.responses.JSONResponse(content=exc_str,
                                              status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY)

    def remove(self, thing, id):
        thing = self.select(thing, id=id, _single=True)
        thing.ghost = True
        self.session.add(thing)
        self.session.commit()
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! delete thing')

    def actually_remove(self, thing, id):
        thing = self.select(thing, id=id, _single=True)
        thing.clear(self)
        self.session.delete(thing)
        self.session.commit()

    def getattr(self, thing, id: UUID, attr):
        cls = self.__backend_models__[thing]
        thing = self.session.exec(sqlmodel.select(cls).where(cls.id == id)).one()
        if not thing: raise ValueErrors(f'no {cls} id {id} found in database')
        thingattr = getattr(thing, attr)
        # if thingattr is None: raise AttributeError(f'db {cls} attr {attr} is None in instance {repr(thing)}')
        return thingattr

    async def setattr(self, request: fastapi.Request, thing: str, id: UUID, attr: str):
        cls = self.__backend_models__[thing]
        thing = self.session.exec(sqlmodel.select(cls).where(cls.id == id)).one()
        if not thing: raise ValueErrors(f'no {cls} id {id} found in database')
        assert thing.model_fields[attr].annotation in (int, float, str)
        body = (await request.body()).decode()
        # print(type(thing), body)
        setattr(thing, attr, body)
        self.session.add(thing)
        self.session.commit()

    def select(self, cls, _count: bool = False, _single=False, user=None, _ghost=False, **kw):
        # print('select', cls, kw)
        if isinstance(cls, str): cls = self.__backend_models__[cls]
        selection = sqlalchemy.func.count(cls.id) if _count else cls
        statement = sqlmodel.select(selection)
        for k, v in kw.items():
            op = operator.eq
            if k.endswith('not'): k, op = k[:-3], operator.ne
            if k.endswith('id') and isinstance(v, str): v = UUID(v)
            if v is not None:
                # print('select where', cls, k, v)
                statement = statement.where(op(getattr(cls, k), v))
        if user: statement = statement.where(getattr(cls, 'userid') == self.user(dict(name=user)).id)
        if not _ghost: statement = statement.where(getattr(cls, 'ghost') == False)
        # print(statement)
        # if statement._get_embedded_bindparams():
        # print({p.key: p.value for p in statement._get_embedded_bindparams()})
        result = self.session.exec(statement)
        try:
            if _count: return int(result.one())
            elif _single: return result.one()
            else: return list(result)
        except sqlalchemy.exc.NoResultFound as e:
            return self.handle_error(e)

    def fix_date(self, x):
        if hasattr(x, 'datecreated') and isinstance(x.datecreated, str):
            x.datecreated = datetime.strptime(x.datecreated, ipd.DATETIME_FORMAT)
        if hasattr(x, 'enddate') and isinstance(x.enddate, str):
            x.enddate = datetime.strptime(x.enddate, ipd.DATETIME_FORMAT)

    def validate_and_add_to_db(self, thing) -> typing.Optional[str]:
        self.fix_date(thing)
        thing = thing.validated_with_backend(self)
        self.session.add(thing)
        try:
            self.session.commit()
            return str(thing.id)
        except sqlalchemy.exc.IntegrityError as e:
            self.session.rollback()
            return self.handle_error(e)

def BACKEND(backendcls, backend_models):
    '''
    Autogen getter methods. Yes, this is better than lots of boilerplate functions that must be kept
    in sync. Any name or name suffixed with 's'
    that is in clientmodels, above, will get /name from the server and turn the result(s) into
    the appropriate client model type, list of such types for plural, or None.
    for _name, _cls in backend_models.items():
    '''
    for _name, _cls in backend_models.items():

        def MAKEMETHOD(name=_name, cls=_cls):
            def create(self, model: dict) -> typing.Union[str, int]:
                # model = cls.parse_obj(model)
                if isinstance(model, dict): model = cls(**model)
                return self.validate_and_add_to_db(model)

            def new(self, **kw) -> typing.Union[str, int]:
                for k, v in kw.copy().items():
                    if k in backend_models:
                        kw[f'{k}id'] = v.id
                        del kw[k]
                model = backend_models[name](**kw)
                newid = getattr(self, f'create_{name}')(model)
                if self.iserror(newid): return newid
                return getattr(self, f'i{name}')(newid, _ghost=True)

            def count(self, kw=None, request: fastapi.Request = None, response_model=int):
                # print('route', name, cls, kw, request, flush=True)
                if request: return self.select(cls, _count=True, **request.query_params)
                elif kw: return self.select(cls, _count=True, **kw)
                else: return self.select(cls, _count=True)

            def multi(self, kw=None, request: fastapi.Request = None, response_model=list[cls]):
                # print('route', name, cls, kw, request, flush=True)
                if request: return self.select(cls, **request.query_params)
                elif kw: return self.select(cls, **kw)
                else: return self.select(cls)

            def single(self, kw=None, request: fastapi.Request = None, response_model=typing.Optional[cls]):
                # print('route', name, cls, kw, request, flush=True)
                if request: return self.select(cls, _single=True, **request.query_params)
                elif kw: return self.select(cls, _single=True, **kw)
                else: return self.select(cls, _single=True)

            def singleid(self, id: str, **kw) -> typing.Union[cls, None]:
                assert id
                return self.select(cls, id=id, _single=True, **kw)

            return multi, single, singleid, count, create, new

        multi, single, singleid, count, create, new = MAKEMETHOD()
        setattr(backendcls, _name, single)
        setattr(backendcls, f'i{_name}', singleid)
        setattr(backendcls, f'{_name}s', multi)
        setattr(backendcls, f'n{_name}s', count)
        setattr(backendcls, f'new{_name}', new)
        if not hasattr(backendcls, f'create_{_name}'):
            setattr(backendcls, f'create_{_name}', create)

Props = list[str]
Attrs = dict[str, typing.Union[str, int, float]]
props_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=list)
attrs_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=dict)

def allfields(model):
    fields = {}
    for base in reversed(model.__mro__):
        if hasattr(base, 'model_fields'):
            fields |= base.model_fields
    return fields

def copy_model(cls, clsname):
    fields = {}
    for name, f in cls.model_fields.items():
        fields[name] = (f.annotation, pydantic.Field(**{k: getattr(f, k) for k in pydantic_field_args}))
    return pydantic.create_model(clsname, **fields)

def make_backend_models(spec_models, SQLModel=sqlmodel.SQLModel):
    print('make_backend_models')
    BackendModelBase = make_backend_model_base(SQLModel)
    body = {kind: {} for kind in spec_models}
    anno = {kind: {} for kind in spec_models}
    remove_fields = {kind: [] for kind in spec_models}
    links, linktables = defaultdict(list), {}
    annospec = {kind: ipd.dev.get_all_annotations(spec) for kind, spec in spec_models.items()}
    specnames = [cls.__name__ for cls in spec_models.values()]
    dbclsname = {kind: 'DB' + cls.__name__.replace('Spec', '') for kind, cls in spec_models.items()}
    dbclsname |= {cls.__name__: 'DB' + cls.__name__.replace('Spec', '') for cls in spec_models.values()}
    specns = {cls.__name__: cls for cls in spec_models.values()}
    for kind, spec in spec_models.items():
        body[kind]['props'] = props_default()
        body[kind]['attrs'] = attrs_default()
        anno[kind]['props'] = Props
        anno[kind]['attrs'] = Attrs
        for attr, field in spec.model_fields.items():
            # if attr == 'name':
            # print(kind, attr, field.annotation, field.metadata)
            if field.metadata == ['UNIQUE']:
                body[kind][attr] = sqlmodel.Field(sa_column_kwargs={'unique': True})
                anno[kind][attr] = field.annotation
            elif len(field.metadata) == 2 and isinstance(field.metadata[0], pydantic.BeforeValidator):
                # ModelRefs id
                assert attr.endswith('id')
                refname, link = field.metadata[1], f'{kind}s'
                if isinstance(refname, tuple): refname, link = refname
                refname = dbclsname[refname]
                optional = field.default is None
                idanno = typing.Optional[UUID] if optional else UUID
                refanno = typing.Optional[refname] if optional else refname
                reffield = sqlmodel.Relationship(back_populates=f'{link}')
                if dbclsname[kind] == refname:
                    # self reference needs direction specified
                    kw = dict(cascade="all", remote_side=f'{dbclsname[kind]}.id')
                idfield = sqlmodel.Field(foreign_key=f'{refname.lower()}.id')
                if optional:
                    idfield = sqlmodel.Field(foreign_key=f'{refname.lower()}.id', nullable=True, default=None)
                body[kind][attr] = idfield
                anno[kind][attr] = idanno
                body[kind][attr[:-2]] = reffield
                anno[kind][attr[:-2]] = refanno
                remote_props[kind].append(attr[:-2])
                # print(kind, attr, anno[kind][attr], 'foreign_key', refname.lower() + '.id')
                # print(kind, attr[:-2], anno[kind][attr[:-2]], 'Rel. backpop', link)
                refkind = refname.replace('DB', '').lower()
                refspec = spec_models[refkind]
                if link in refspec.model_fields:
                    # print('    link', refkind, link, dbclsname[kind])
                    refmeta = refspec.model_fields[link].metadata
                    if len(refmeta) != 2 or refmeta[1] != spec.__name__:
                        err = f'{kind}.{attr} linked {refkind}.{link} must be list["{refspec.__name__}"]'
                        raise TypeError(err)
                # print(refkind, link, list[dbclsname[kind]], 'Rel. backpop', attr[:-2])
                body[refkind][link] = sqlmodel.Relationship(back_populates=attr[:-2])
                anno[refkind][link] = list[dbclsname[kind]]
            elif hasattr(field.annotation, '__origin__') and field.annotation.__origin__ == dict:
                body[kind][attr] = attrs_default()
                anno[kind][attr] = Attrs
            elif hasattr(field.annotation, '__origin__') and field.annotation.__origin__ == list:
                args = typing.get_args(field.annotation)
                # print(kind, attr, args)
                if args[0] in specnames or args[0] in spec_models.values():
                    assert len(args) < 3
                    if len(args) == 2: refname, link = args
                    else: refname, link = args[0].__name__, 'DEFAULT'
                    if not isinstance(refname, str): refname = refname.__name__
                    links[tuple(sorted([spec.__name__, refname])), link].append((kind, attr, refname))
                    remove_fields[kind].append(attr)

    # create many to many relations
    for (pair, link), ends in links.items():
        assert len(ends) == 2, f'links of order > 2 not supported, {pair}, {link}'
        linkname = f'{str.join("",pair).replace("Spec", "")}{link.title()}Link'
        (kind1, attr1, refname1), (kind2, attr2, refname2) = ends
        assert refname1 in pair and refname2 in pair
        linkbody = {
            f'{kind1}id': sqlmodel.Field(default=None, primary_key=True, foreign_key=f'db{kind1}.id'),
            f'{kind2}id': sqlmodel.Field(default=None, primary_key=True, foreign_key=f'db{kind2}.id'),
        }
        linkanno = {f'{kind1}id': typing.Optional[UUID], f'{kind2}id': typing.Optional[UUID]}
        linkbody['__annotations__'] = linkanno
        # rich.print(linkbody)
        linkcls = type(linkname, (SQLModel, ), linkbody, table=True)
        linktables[linkname] = linkcls
        kinds, attrs, refnames = zip(*ends)
        for i, (kind, attr, refname) in enumerate(zip(kinds, attrs, refnames)):
            otherkind, otherattr = kinds[not i], attrs[not i]
            kw = dict(back_populates=otherattr, link_model=linkcls)
            if pair[0] == pair[1]:
                kw['sa_relationship_kwargs'] = dict(
                    primaryjoin=f'{dbclsname[kind]}.id=={linkname}.{attr}id',
                    secondaryjoin=f'{dbclsname[kind]}.id=={linkname}.{otherattr}id',
                )
            body[kind][attr] = sqlmodel.Relationship(**kw)
            anno[kind][attr] = list[dbclsname[otherkind]]
            # print(f'{kind} {attr} : {anno[kind][attr]} = Relationship({kw})')

    # build the models
    models = {}
    for kind, origspec in spec_models.items():
        spec = copy_model(origspec, f'{origspec.__name__}Trim')
        attrs = set(spec_models[kind].model_fields) - set(body[kind])
        attrs = attrs - {'ispublic', 'ghost', 'datecreated', 'telemetry', 'id'}
        # print(kind, attrs)
        # for attr in body[kind]:
        # print(f'    {attr:12} {str(anno[kind][attr]):20} {body[kind][attr]}')
        for name, member in spec.__dict__.copy().items():
            if hasattr(member, '__layer__') and member.__layer__ == 'backend':
                body[kind][name] = member
                delattr(spec, name)
        body[kind]['__annotations__'] = anno[kind]
        for attr in remove_fields[kind] + list(body[kind]):
            if attr in spec.model_fields: del spec.model_fields[attr]
            if hasattr(spec, attr): delattr(spec, attr)
        # print(spec.model_fields.keys())
        models[kind] = type(dbclsname[kind], (BackendModelBase, spec), body[kind], table=True)
        print('spec diff', kind, origspec.model_fields.keys())
        print('spec diff', kind, spec.model_fields.keys())

    remote_props = {}
    for kind in spec_models:
        remote_props[kind] = {k for k, v in body[kind].items() if isinstance(v, RelationshipInfo)}

    return models, linktables, remote_props

if False:
    filekindid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbfilekind.id', nullable=True)
    filekind: Optional[DBFileKind] = Relationship(back_populates='pollfiles')
    parentid: Optional[uuid.UUID] = Field(default=None, foreign_key='dbpollfile.id', nullable=True)
    parent: Optional['DBPollFile'] = Relationship(back_populates='children',
                                                  sa_relationship_kwargs=dict(cascade="all",
                                                                              remote_side='DBPollFile.id'))
    workflowid: uuid.UUID = Field(foreign_key='dbworkflow.id', nullable=True)
    flowstepid: uuid.UUID = Field(default=None, foreign_key='dbflowstep.id')
    workflowid: uuid.UUID = Field(foreign_key='dbworkflow.id')

    Relationship(back_populates='cmds', link_model=DBPymolCMDFlowStepLink)
    Relationship(back_populates='filekind')
    Relationship(
        back_populates='following',
        link_model=DBUserUserLink,
        sa_relationship_kwargs=dict(
            primaryjoin="DBUser.id==DBUserUserLink.followerid",
            secondaryjoin="DBUser.id==DBUserUserLink.followingid",
        ),
    )
    Relationship(
        back_populates='followers',
        link_model=DBUserUserLink,
        sa_relationship_kwargs=dict(
            primaryjoin="DBUser.id==DBUserUserLink.followingid",
            secondaryjoin="DBUser.id==DBUserUserLink.followerid",
        ),
    )

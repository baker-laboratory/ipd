from collections import defaultdict
import contextlib
from datetime import datetime
import fastapi
import ipd
from ipd.crud.frontend import SpecBase
import operator
import pydantic
import sqlalchemy
import sqlmodel.pool
from sqlmodel.main import RelationshipInfo
import sys
import traceback
import typing
from uuid import UUID, uuid4

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

def make_backend_model_base(SQL=sqlmodel.SQLModel):
    class BackendModelBase(SQL, SpecBase):
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

class BackendBase:
    def __init_subclass__(cls, models, SQL=sqlmodel.SQLModel, **kw):
        super().__init_subclass__(**kw)
        cls.__spec_models__ = models
        backend_info = ipd.crud.backend.make_backend_models(cls, SQL)
        cls.__backend_models__, cls.__remoteprops__, cls.__trimspecs__ = backend_info
        cls.__spec_models__ = {name: mdl.__spec__ for name, mdl in cls.__backend_models__.items()}
        add_basic_backend_model_methods(cls)
        for dbcls in cls.__backend_models__.values():
            setattr(cls, dbcls.__name__, dbcls)

    def __init__(self, engine):
        memkw = dict(connect_args={"check_same_thread": False}, poolclass=sqlmodel.pool.StaticPool)
        if engine == '<memory>': engine = sqlmodel.create_engine('sqlite://', **memkw)
        if isinstance(engine, str) and '://' not in engine: engine = f'sqlite:///{engine}'
        if isinstance(engine, str): engine = engine = sqlmodel.create_engine(engine)
        self.engine = engine
        self.session = sqlmodel.Session(self.engine)
        self.router = fastapi.APIRouter()
        self.app = fastapi.FastAPI()
        self.route = self.router.add_api_route
        for model in self.__backend_models__:
            self.route(f'/api/{model}', getattr(self, model), methods=['GET'])
            self.route(f'/api/{model}s', getattr(self, f'{model}s'), methods=['GET'])
            self.route(f'/api/n{model}s', getattr(self, f'n{model}s'), methods=['GET'])
            self.route(f'/api/create/{model}', getattr(self, f'create_{model}'), methods=['POST'])
        self.route('/api/getattr/{kind}/{id}/{attr}', self.getattr, methods=['GET'])
        self.route('/api/setattr/{kind}/{id}/{attr}', self.setattr, methods=['POST'])
        self.route('/api/remove/{kind}/{id}', self.remove, methods=['GET'])
        self.route('/api/remove/{kind}/{id}', self.remove, methods=['GET'])
        self.route('/api/gitstatus', ipd.dev.git_status, methods=['GET'])
        self.route('/api/gitstatus/{header}/{footer}', ipd.dev.git_status, methods=['GET'])
        self.init_routes()
        self.app.include_router(self.router)
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

    def init_routes(self):
        pass

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

    def remove(self, kind, id):
        thing = self.select(kind, id=id, _single=True)
        thing.ghost = True
        self.session.add(thing)
        self.session.commit()
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! delete thing')

    def actually_remove(self, kind, id):
        thing = self.select(kind, id=id, _single=True)
        thing.clear(self)
        self.session.delete(thing)
        self.session.commit()

    def getattr(self, kind, id: UUID, attr):
        cls = self.__backend_models__[kind]
        thing = self.session.exec(sqlmodel.select(cls).where(cls.id == id)).one()
        if not thing: raise ValueErrors(f'no {cls} id {id} found in database')
        thingattr = getattr(thing, attr)
        # if thingattr is None: raise AttributeError(f'db {cls} attr {attr} is None in instance {repr(thing)}')
        return thingattr

    async def setattr(self, request: fastapi.Request, kind: str, id: UUID, attr: str):
        cls = self.__backend_models__[kind]
        thing = self.session.exec(sqlmodel.select(cls).where(cls.id == id)).one()
        if not thing: raise ValueErrors(f'no {cls} id {id} found in database')
        if attr in thing.model_fields:
            assert thing.model_fields[attr].annotation in (int, float, str, UUID)
        body = (await request.body()).decode()
        with contextlib.suppress((AttributeError, ValueError)):
            body = UUID(body)
        try:
            body = [UUID(id) for id in body[1:-1].split(',')]
            body = list(map(getattr(self, f'i{kind}'), body))
            setattr(thing, attr, body)
        except (ValueError, TypeError):
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

def add_basic_backend_model_methods(backendcls):
    '''
    Autogen getter methods. Yes, this is better than lots of boilerplate functions that must be kept
    in sync. Any name or name suffixed with 's'
    that is in clientmodels, above, will get /name from the server and turn the result(s) into
    the appropriate client model type, list of such types for plural, or None.
    '''
    for _name, _dbcls in backendcls.__backend_models__.items():

        def make_basic_backend_model_methods_closure(name=_name, dbcls=_dbcls):
            def create(self, model: dict) -> typing.Union[str, int]:
                # model = dbcls.parse_obj(model)
                if isinstance(model, dict): model = dbcls(**model)
                return self.validate_and_add_to_db(model)

            def new(self, **kw) -> typing.Union[str, int]:
                for k, v in kw.copy().items():
                    if k in dbcls.__sibling_models__:
                        kw[f'{k}id'] = v.id
                        del kw[k]
                model = dbcls(**kw)
                newid = getattr(self, f'create_{name}')(model)
                if self.iserror(newid): return newid
                return getattr(self, f'i{name}')(newid, _ghost=True)

            def count(self, kw=None, request: fastapi.Request = None, response_model=int):
                # print('route', name, dbcls, kw, request, flush=True)
                if request: return self.select(dbcls, _count=True, **request.query_params)
                elif kw: return self.select(dbcls, _count=True, **kw)
                else: return self.select(dbcls, _count=True)

            def multi(self, kw=None, request: fastapi.Request = None, response_model=list[dbcls]):
                # print('route', name, dbcls, kw, request, flush=True)
                if request: return self.select(dbcls, **request.query_params)
                elif kw: return self.select(dbcls, **kw)
                else: return self.select(dbcls)

            def single(self, kw=None, request: fastapi.Request = None, response_model=typing.Optional[dbcls]):
                # print('route', name, dbcls, kw, request, flush=True)
                if request: return self.select(dbcls, _single=True, **request.query_params)
                elif kw: return self.select(dbcls, _single=True, **kw)
                else: return self.select(dbcls, _single=True)

            def bcount(self, count=count, **kw) -> int:
                return count(self, kw)

            def bmulti(self, multi=multi, **kw) -> list[dbcls]:
                return multi(self, kw)

            def bsingle(self, single=single, **kw) -> typing.Optional[dbcls]:
                return single(self, kw)

            def singleid(self, id: str, **kw) -> typing.Union[dbcls, None]:
                assert id
                return self.select(dbcls, id=id, _single=True, **kw)

            return multi, single, singleid, count, create, new, bcount, bmulti, bsingle

        funcs = make_basic_backend_model_methods_closure()
        multi, single, singleid, count, create, new, bcount, bmulti, bsingle = funcs
        singleid.__qualname__ = f'{backendcls.__name__}.i{_name}'
        single.__qualname__ = f'{backendcls.__name__}.{_name}'
        multi.__qualname__ = f'{backendcls.__name__}.{_name}s'
        count.__qualname__ = f'{backendcls.__name__}.n{_name}s'
        bsingle.__qualname__ = f'{backendcls.__name__}.b{_name}'
        bmulti.__qualname__ = f'{backendcls.__name__}.b{_name}s'
        bcount.__qualname__ = f'{backendcls.__name__}.bn{_name}s'
        new.__qualname__ = f'{backendcls.__name__}.new{_name}'
        create.__qualname__ = f'{backendcls.__name__}.create_{_name}'
        setattr(backendcls, f'i{_name}', singleid)
        setattr(backendcls, _name, single)
        setattr(backendcls, f'{_name}s', multi)
        setattr(backendcls, f'n{_name}s', count)
        setattr(backendcls, f'b{_name}', bsingle)
        setattr(backendcls, f'b{_name}s', bmulti)
        setattr(backendcls, f'bn{_name}s', bcount)
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

def copy_model_cls(cls, clsname) -> typing.Type[pydantic.BaseModel]:
    fields, funcs = {}, {}
    for name, f in cls.model_fields.items():
        fields[name] = (f.annotation, pydantic.Field(**{k: getattr(f, k) for k in pydantic_field_args}))
    newcls = pydantic.create_model(clsname, __validators__=funcs, __base__=cls.__bases__, **fields)
    # newcls.__annotations__ = cls.__annotations__
    return newcls

def make_backend_models(backendcls, SQL=sqlmodel.SQLModel):
    spec_models = backendcls.__spec_models__
    # print('make_backend_models')
    BackendModelBase = make_backend_model_base(SQL)
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
        assert kind == spec.__name__[:-4].lower()
        # print('make_backend_models', kind, spec.__name__)
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
                kw = {}
                if dbclsname[kind] == refname:
                    kw = {'sa_relationship_kwargs': dict(cascade="all", remote_side=f'{dbclsname[kind]}.id')}
                reffield = sqlmodel.Relationship(back_populates=f'{link}', **kw)
                idfield = sqlmodel.Field(foreign_key=f'{refname.lower()}.id')
                if optional:
                    idfield = sqlmodel.Field(foreign_key=f'{refname.lower()}.id', nullable=True, default=None)
                body[kind][attr] = idfield
                anno[kind][attr] = typing.Optional[UUID] if optional else UUID
                body[kind][attr[:-2]] = reffield
                anno[kind][attr[:-2]] = typing.Optional[refname] if optional else refname
                # if attr == 'pollid':
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
                    elif isinstance(args[0], str): refname, link = args[0], 'DEFAULT'
                    else: refname, link = args[0].__name__, 'DEFAULT'
                    if not isinstance(refname, str): refname = refname.__name__
                    links[tuple(sorted([spec.__name__, refname])), link].append((kind, attr, refname))
                    remove_fields[kind].append(attr)

    # create many to many relations
    for (pair, link), ends in links.items():
        assert len(ends) == 2, f'links of order > 2 not supported, {pair}, {link}'
        linkname = f'{str.join("",pair).replace("Spec", "")}{link.title()}Link'
        (kind1, attr1, refname1), (kind2, attr2, refname2) = ends
        # print(pair, link, ends)
        assert refname1 in pair and refname2 in pair
        linkbody = {
            f'{attr1}id': sqlmodel.Field(default=None, primary_key=True, foreign_key=f'db{kind1}.id'),
            f'{attr2}id': sqlmodel.Field(default=None, primary_key=True, foreign_key=f'db{kind2}.id'),
        }
        linkanno = {f'{attr1}id': typing.Optional[UUID], f'{attr2}id': typing.Optional[UUID]}
        linkbody['__annotations__'] = linkanno
        # rich.print(linkbody)
        # print(kind1, kind2)
        linkcls = type(linkname, (SQL, ), linkbody, table=True)
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
    backend_models, remote_props, trimspecs = {}, {}, {}
    for kind, spec in spec_models.items():
        for name, member in spec.__dict__.copy().items():
            if hasattr(member, '__layer__') and member.__layer__ == 'backend':
                body[kind][name] = member.__wrapped__
                delattr(spec, name)
        trimspec = copy_model_cls(spec, f'{spec.__name__}Trim')
        attrs = set(spec_models[kind].model_fields) - set(body[kind])
        attrs = attrs - {'ispublic', 'ghost', 'datecreated', 'telemetry', 'id', '__annotations__'}
        for attr in remove_fields[kind] + list(body[kind]):
            # if attr[0] == '_': continue
            # print(kind, 'remove', attr)
            ano = trimspec.__annotations__.get(attr, None)
            ano = ano or trimspec.model_fields.get(attr, None)
            ano = getattr(ano, '__origin__', None)
            if ano == list:
                if attr in trimspec.model_fields: del trimspec.model_fields[attr]
                if hasattr(trimspec, attr): delattr(trimspec, attr)
                if attr in trimspec.__annotations__: del trimspec.__annotations__[attr]

        body[kind]['__annotations__'] = anno[kind]
        dbcls = type(dbclsname[kind], (BackendModelBase, trimspec), body[kind], table=True)
        dbcls.__spec__ = spec
        spec.__backend_model__ = dbcls
        dbcls.__sibling_models__ = backend_models
        backend_models[kind] = dbcls
        trimspecs[kind] = trimspec
        remote_props[kind] = {k for k, v in body[kind].items() if isinstance(v, RelationshipInfo)}
        setattr(sys.modules[backendcls.__module__], dbcls.__name__, dbcls)

    # update namespace of @backend_method functions
    for dbcls in backend_models.values():
        for name, member in dbcls.__dict__.items():
            if hasattr(member, '__layer__') and member.__layer__ == 'backend':
                member.__module__ = backendcls.__module__
                for dbcls2 in backend_models.values():
                    member.__globals__[dbcls2.__name__] = dbcls2
                    assert dbcls2.__name__ in getattr(dbcls, name).__globals__

    return backend_models, remote_props, trimspecs

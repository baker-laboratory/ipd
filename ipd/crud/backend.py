import asyncio
import contextlib
import operator
import sys
import traceback
import typing
from collections import defaultdict
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import fastapi
import pydantic
import sqlalchemy
import sqlmodel.pool
from sqlalchemy.orm import registry
from sqlmodel.main import RelationshipInfo

import ipd
from ipd.crud.frontend import SpecBase

# profiler= ipd.dev.timed
profiler = lambda f: f

backend_type_map = {
    pydantic.AnyUrl: str,
    pydantic.FilePath: str,
    pydantic.NewPath: str,
    pydantic.DirectoryPath: str,
}

Props = list[str]
Attrs = dict[str, typing.Union[str, int, float]]
list_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=list)
dict_default = lambda: sqlmodel.Field(sa_column=sqlalchemy.Column(sqlalchemy.JSON), default_factory=dict)
sqlmodel_field_map = {Props: list_default, Attrs: dict_default}
pydantic_field_args = ('default default_factory alias alias_priority validation_alias serialization_alias '
                       'title field_title_generator description examples exclude discriminator deprecated '
                       'json_schema_extra frozen validate_default repr init init_var kw_only').split()

def to_sqlalchemy_types(thing):
    # if isinstance(thing, pydantic.fields.FieldInfo):
    #     print('anno', thing.annotation)
    #     if thing.annotation in sqlmodel_field_map:
    #         return sqlmodel_field_map[thing.thing]
    if isinstance(thing, list): return [to_sqlalchemy_types(x) for x in thing]
    if isinstance(thing, dict): return {n: to_sqlalchemy_types(x) for n, x in thing.items()}
    if thing in backend_type_map: return backend_type_map[thing]
    return thing

def allfields(model):
    fields = {}
    for base in reversed(model.__mro__):
        if hasattr(base, 'model_fields'):
            fields |= base.model_fields
    return fields

@profiler
def copy_model_cls(cls, clsname, renamed_attrs) -> type[pydantic.BaseModel]:
    fields, funcs = {}, {}
    assert all(not hasattr(cls, newattr) for newattr in renamed_attrs.values())
    for name, f in cls.model_fields.items():
        if name in renamed_attrs: name = renamed_attrs[name]
        if f.annotation in sqlmodel_field_map:
            fields[name] = (f.annotation, sqlmodel_field_map[f.annotation]())
        else:
            newfield = pydantic.Field(**{k: getattr(f, k) for k in pydantic_field_args})
            fields[name] = (to_sqlalchemy_types(f.annotation), newfield)
    newcls = pydantic.create_model(clsname, __validators__=funcs, __base__=cls.__bases__, **fields)
    # newcls.__annotations__ = cls.__annotations__
    return newcls

class BackendError(Exception):
    pass

@profiler
def make_backend_model_base(SQL):
    class BackendModelBase(SQL, SpecBase):
        id: UUID = sqlmodel.Field(primary_key=True, default_factory=uuid4)

        def clear(self, backend, ghost=True):
            return

        def validated_with_backend(self, backend):
            for name, field in self.model_fields.items():
                # print(field.annotation, self[name], type(self[name]))
                if field.annotation in (UUID, Optional[UUID]):
                    # print(name, self[name])
                    if name == 'id' and self[name] is None: self[name] = uuid4()
                    elif isinstance(self[name], str):
                        try:
                            self[name] = UUID(self[name])
                        except ValueError as e:
                            raise ValueError(f'bad UUID string "{self[name]}"') from e
                            # print('success')
            return self

    return BackendModelBase

def process_specs(specs):
    for kind, spec in specs.items():
        for k, v in spec.model_fields.items():
            if k != 'exe': continue
            # print(k, v.annotation, v.metadata, v)
            # print(dir(v))
    # assert 0
    return specs

@profiler
class BackendBase:
    mountpoint = 'ipd'

    def __init_subclass__(cls, models: dict[str, SpecBase], SQL=sqlmodel.SQLModel, **kw):
        super().__init_subclass__(**kw)
        cls.__sqlmodel__ = SQL
        cls._BackendModelBase = make_backend_model_base(SQL)
        if isinstance(models, list): models = {m.modelkind(): m for m in models}  # type: ignore
        cls.__spec_models__ = process_specs(models)
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
        self.route('/api/setattr/{kind}/{id}/{attr}/{attrkind}', self.setattr, methods=['POST'])
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
            result = self.session.exec(statement)  # type: ignore
            self.session.commit()
        self.initdb()

    def add_defaults(self):
        pass

    def initdb(self):
        self.__sqlmodel__.metadata.create_all(self.engine)

    def init_routes(self):
        pass

    def commit(self):
        self.session.commit()

    def iserror(self, thing):
        return isinstance(thing, fastapi.responses.Response) and thing.status_code != 200

    def handle_error(self, exc):
        exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
        print('!' * 80)
        print(exc_str)
        print('!' * 80)
        result = f'{"!"*80}\n{exc_str}\nSTACK:\n{traceback.format_exc()}\n{"!"*80}'
        return fastapi.responses.JSONResponse(content=exc_str, status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY)

    def remove(self, kind, id):
        thing = self.select(kind, id=id, _single=True)
        thing.ghost = True  # type: ignore
        self.session.add(thing)
        self.session.commit()
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! delete thing')

    def actually_remove(self, kind, id):
        thing = self.select(kind, id=id, _single=True)
        thing.clear(self)  # type: ignore
        self.session.delete(thing)
        self.session.commit()

    def getattr(self, kind: str, id: UUID, attr: str):
        cls = self.__backend_models__[kind]
        thing = self.session.exec(sqlmodel.select(cls).where(cls.id == id)).one()
        if not thing: raise ValueErrors(f'no {cls} id {id} found in database')  # type: ignore
        thingattr = getattr(thing, attr)
        if hasattr(thingattr, 'modelkind'):
            table = self.__sqlmodel__.metadata.tables[f'db{thingattr.modelkind()}']
            for k, v in thingattr.model_fields.items():
                satype = table.columns[k].type.__class__.__name__
                if satype == 'JSON' and isinstance(thingattr[k], str):
                    thingattr[k] = ipd.dev.str_to_json(thingattr[k])
        return thingattr

    def setattr(self, request: fastapi.Request, kind: str, id: UUID, attr: str, attrkind: str = ''):
        cls = self.__backend_models__[kind]
        thing = self.session.exec(sqlmodel.select(cls).where(cls.id == id)).one()
        if not thing: raise ValueErrors(f'no {cls} id {id} found in database')  # type: ignore
        body = asyncio.run(request.body()).decode()
        # body = asyncio.run(request.body()).decode()
        with contextlib.suppress((AttributeError, ValueError)):  # type: ignore
            body = UUID(body)
        if attrkind:
            body = [] if body == '[]' else body[1:-1].split(',')  # type: ignore
            body = [UUID(id) for id in body]
            body = [coro for coro in list(map(getattr(self, f'i{attrkind}'), body))]
            setattr(thing, attr, body)
        else:
            if attr in thing.model_fields:
                oktypes = (int, float, str, UUID, list[int], list[float], list[str], list[UUID])
                assert thing.model_fields[attr].annotation in oktypes
            setattr(thing, attr, body)
        self.session.add(thing)
        self.session.commit()
        # ic(thing, attr, body, getattr(thing, attr))

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
        if user: statement = statement.where(getattr(cls, 'userid') == self.user(dict(name=user)).id)  # type: ignore
        if not _ghost: statement = statement.where(getattr(cls, 'ghost') == False)  # noqa
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

    def validate_and_add_to_db(self, thing) -> Optional[str]:
        self.fix_date(thing)
        thing = thing.validated_with_backend(self)
        self.session.add(thing)
        try:
            self.session.commit()
            return str(thing.id)
        except sqlalchemy.exc.IntegrityError as e:
            self.session.rollback()
            return self.handle_error(e)  # type: ignore

    def fields_name_to_id(self, dbcls: pydantic.BaseModel, modeldict: dict):
        if not isinstance(modeldict, dict): return modeldict
        for k, v in modeldict.copy().items():
            if not isinstance(v, str): continue
            if k == 'name' or k.endswith('id'): continue
            if k in dbcls.model_fields: continue
            # if ipd.dev.touuid(v): continue
            # if ipd.dev.todatetime(v): continue
            anno = typing.get_args(dbcls.__annotations__[k])[0].__forward_arg__
            kind = anno[2:].lower()
            thing = getattr(self, kind)(dict(name=v))
            modeldict[f'{k}id'] = thing.id
            del modeldict[k]
            # ic(k, v, thing.id)
        return modeldict

def fields_uuidstr_to_id(vals):
    if not isinstance(vals, dict): return vals
    for k, v in vals.copy().items():
        if not k.endswith('id') and (uid := ipd.dev.touuid(v)):
            del vals[k]
            vals[f'{k}id'] = uid
    return vals

def add_basic_backend_model_methods(backendcls):
    """Autogen getter methods.

    Yes, this is better than lots of boilerplate functions that must be
    kept in sync. Any name or name suffixed with 's' that is in
    clientmodels, above, will get /name from the server and turn the
    result(s) into the appropriate client model type, list of such types
    for plural, or None.
    """
    for _name, _dbcls in backendcls.__backend_models__.items():

        def make_basic_backend_model_methods_closure(name=_name, dbcls=_dbcls):
            def create(self, model: dict) -> typing.Union[str, int]:
                # model = dbcls.parse_obj(model)
                assert isinstance(self, BackendBase)
                model = fields_uuidstr_to_id(model)
                model = self.fields_name_to_id(dbcls, model)
                if isinstance(model, dict): model = dbcls(**model)
                return self.validate_and_add_to_db(model)  # type: ignore

            def new(self, **kw) -> typing.Union[str, int]:
                assert isinstance(self, BackendBase)
                for k, v in kw.copy().items():
                    if k in dbcls.__sibling_models__:
                        kw[f'{k}id'] = v.id
                        del kw[k]
                model = dbcls(**kw)
                newid = (getattr(self, f'create_{name}')(model))
                if self.iserror(newid): return newid
                return (getattr(self, f'i{name}')(newid, _ghost=True))

            def count(self, kw=None, request: fastapi.Request = None, response_model=int):  # type: ignore
                # print('route', name, dbcls, kw, request, flush=True)
                assert isinstance(self, BackendBase)
                if request: return self.select(dbcls, _count=True, **request.query_params)  # type: ignore
                elif kw: return self.select(dbcls, _count=True, **kw)
                else: return self.select(dbcls, _count=True)

            def single(self, kw=None, request: fastapi.Request = None, response_model=Optional[dbcls]):  # type: ignore
                # print('route', name, dbcls, kw, request, flush=True)
                assert isinstance(self, BackendBase)
                if request: return self.select(dbcls, _single=True, **request.query_params)  # type: ignore
                elif kw: return self.select(dbcls, _single=True, **kw)
                else: return self.select(dbcls, _single=True)

            def multi(self, kw=None, request: fastapi.Request = None, response_model=list[dbcls]):  # type: ignore
                # print('route', name, dbcls, kw, request, flush=True)
                # ipd.dev.global_timer.checkpoint('multi')
                assert isinstance(self, BackendBase)
                if request: return self.select(dbcls, **request.query_params)  # type: ignore
                elif isinstance(kw, list): return [single(self, dict(name=n)) for n in kw]
                elif kw: return self.select(dbcls, **kw)
                else: return self.select(dbcls)

            def bcount(self, count=count, **kw) -> int:
                assert isinstance(self, BackendBase)
                return count(self, kw)

            def bmulti(self, multi=multi, **kw) -> list[dbcls]:  # type: ignore
                assert isinstance(self, BackendBase)
                return multi(self, kw)

            def bsingle(self, single=single, **kw) -> Optional[dbcls]:  # type: ignore
                assert isinstance(self, BackendBase)
                return single(self, kw)

            def singleid(self, id: str, **kw) -> typing.Union[dbcls, None]:  # type: ignore
                assert isinstance(self, BackendBase)
                assert id
                return self.select(dbcls, id=id, _single=True, **kw)

            return create, {
                f'i{_name}': singleid,
                _name: single,
                f'{_name}s': multi,
                f'n{_name}s': count,
                f'b{_name}': bsingle,
                f'b{_name}s': bmulti,
                f'bn{_name}s': bcount,
                f'new{_name}': new,
            }
            # return multi, single, singleid, count, create, new, bcount, bmulti, bsingle

        create, funcs = make_basic_backend_model_methods_closure()
        for attr, fn in funcs.items():
            fn.__qualname__ = f'{backendcls.__name__}.{attr}'
            # setattr(backendcls, attr, fn)
            setattr(backendcls, attr, profiler(fn))

        # funcs = make_basic_backend_model_methods_closure()
        # multi, single, singleid, count, create, new, bcount, bmulti, bsingle = funcs
        # singleid.__qualname__ = f'{backendcls.__name__}.i{_name}'
        # single.__qualname__ = f'{backendcls.__name__}.{_name}'
        # multi.__qualname__ = f'{backendcls.__name__}.{_name}s'
        # count.__qualname__ = f'{backendcls.__name__}.n{_name}s'
        # bsingle.__qualname__ = f'{backendcls.__name__}.b{_name}'
        # bmulti.__qualname__ = f'{backendcls.__name__}.b{_name}s'
        # bcount.__qualname__ = f'{backendcls.__name__}.bn{_name}s'
        # new.__qualname__ = f'{backendcls.__name__}.new{_name}'
        create.__qualname__ = f'{backendcls.__name__}.create_{_name}'
        # setattr(backendcls, f'i{_name}', singleid)
        # setattr(backendcls, _name, single)
        # setattr(backendcls, f'{_name}s', multi)
        # setattr(backendcls, f'n{_name}s', count)
        # setattr(backendcls, f'b{_name}', bsingle)
        # setattr(backendcls, f'b{_name}s', bmulti)
        # setattr(backendcls, f'bn{_name}s', bcount)
        # setattr(backendcls, f'new{_name}', new)
        if not hasattr(backendcls, f'create_{_name}'):
            setattr(backendcls, f'create_{_name}', create)

def check_list_T(kind, spec, attr, annotation, links, specbyname, models, rmfields):
    if isinstance(annotation, tuple):
        args = annotation
    else:
        args = typing.get_args(annotation)
    # print(kind, attr, args)
    if args[0] in specbyname or args[0] in models.values():
        assert len(args) < 3
        if len(args) == 2: refname, link = args
        elif isinstance(args[0], str): refname, link = args[0], 'DEFAULT'  # type: ignore
        else: refname, link = args[0].__name__, 'DEFAULT'  # type: ignore
        # print(kind, attr, link, annotation)
        if not isinstance(refname, str): refname = refname.__name__
        links[tuple(sorted([spec.__name__, refname])), link].append((kind, attr, refname))
        rmfields[kind].append(attr)
        return True
    return False

def field_is_ref(field):
    return (len(field.metadata) == 2 and isinstance(field.metadata[0], pydantic.BeforeValidator))

def field_is_ref_to_list(field):
    return (isinstance(field.metadata[1], tuple) and len(field.metadata[1]) == 2 and isinstance(field.metadata[1][1], str)
            and len(typing.get_args(field.metadata[1][0])))

def newSQL() -> type[sqlmodel.SQLModel]:
    return type('NewBase', (sqlmodel.SQLModel, ), {}, registry=registry())

def make_backend_models(backendcls, SQL=None, debug=False):
    SQL = SQL or newSQL()
    models = backendcls.__spec_models__
    # ic(models['method'].model_fields)
    # print('make_backend_models')
    body = {kind: {} for kind in models}
    anno = {kind: {} for kind in models}
    rmfields = {kind: [] for kind in models}
    keepattrs = {kind: set() for kind in models}
    links, linktables = defaultdict(list), {}
    annospec = {kind: ipd.dev.get_all_annotations(spec) for kind, spec in models.items()}
    specbyname = {cls.__name__: cls for cls in models.values()}
    dbclsname = {kind: 'DB' + cls.__name__.replace('Spec', '') for kind, cls in models.items()}
    dbclsname |= {cls.__name__: 'DB' + cls.__name__.replace('Spec', '') for cls in models.values()}
    specns = {cls.__name__: cls for cls in models.values()}
    renamed_attrs = {kind: {} for kind in models}
    for kind, spec in models.items():
        assert kind == spec.__name__[:-4].lower()
        # print('make_backend_models', kind, spec.__name__)
        for attr, field in spec.model_fields.items():
            if attr == 'name':
                body[kind][attr] = sqlmodel.Field(index=True, sa_column_kwargs={'unique': True})
                anno[kind][attr] = str
            elif hasattr(spec, 'props'):
                body[kind]['props'] = list_default()
                anno[kind]['props'] = Props
            elif hasattr(spec, 'attrs'):
                body[kind]['attrs'] = dict_default()
                anno[kind]['attrs'] = Attrs
                anno[kind][attr] = field.annotation
            elif field.metadata == ['UNIQUE']:
                body[kind][attr] = sqlmodel.Field(sa_column_kwargs={'unique': True})
            elif field_is_ref(field):
                # ModelRefs id
                # assert attr.endswith('id'), f'{spec.__name__} single ModelRef attr "{attr}" doesnt end wtih id'
                if field_is_ref_to_list(field):
                    meta = typing.get_args(field.metadata[1][0])[0], field.metadata[1][1]
                    # print('Ref list', kind, attr, meta)
                    check_list_T(kind, spec, attr, meta, links, specbyname, models, rmfields)
                    body[kind][attr] = None
                    anno[kind][attr] = field.metadata[1][0]
                    continue
                if not attr.endswith('id'):
                    renamed_attrs[kind][attr] = f'{attr}id'
                    assert not hasattr(spec, renamed_attrs[kind][attr])
                    attr = renamed_attrs[kind][attr]
                refname, refattr, tagged = field.metadata[1], f'{kind}s', False
                if isinstance(refname, tuple):
                    # refname, refattr, tagged = refname[0], f'{refattr}{refname[1]}', True
                    refname, refattr, tagged = refname[0], f'{refname[1]}', True
                if not isinstance(refname, str): refname = refname.__name__
                refname = dbclsname[refname]
                optional = field.default is None
                kw = {}
                if dbclsname[kind] == refname:
                    kw = {'sa_relationship_kwargs': dict(remote_side=f'{dbclsname[kind]}.id')}
                elif tagged:
                    kw = {'sa_relationship_kwargs': dict()}
                reffield = sqlmodel.Relationship(back_populates=f'{refattr}', **kw)  # type: ignore
                idfield = sqlmodel.Field(foreign_key=f'{refname.lower()}.id')
                if optional:
                    idfield = sqlmodel.Field(foreign_key=f'{refname.lower()}.id', nullable=True, default=None)
                body[kind][attr] = idfield
                anno[kind][attr] = Optional[UUID] if optional else UUID
                body[kind][attr[:-2]] = reffield
                anno[kind][attr[:-2]] = Optional[refname] if optional else refname

                # print(attr)
                # spec.__annotations__[attr] = UUID
                # spec.__annotations__[attr[-2:]] = Optional[refname]
                # spec.model_fields[attr] = pydantic.Field(annotation=UUID)
                # spec.model_fields[attr[-2:]] = pydantic.Field(annotation=Optional[refname], default=None)

                # if attr == 'pollid':
                # print(kind, attr, anno[kind][attr], 'foreign_key', refname.lower() + '.id')
                # print(kind, attr[:-2], anno[kind][attr[:-2]], 'Rel. backpop', refattr)
                refkind = refname.replace('DB', '').lower()
                refspec = models[refkind]
                if refattr in refspec.model_fields:
                    # print('    refattr', refkind, refattr, dbclsname[kind])
                    refmeta = refspec.model_fields[refattr].metadata
                    if len(refmeta) != 2 or refmeta[1] != spec.__name__:
                        err = f'{kind}.{attr} linked {refkind}.{refattr} must be list["{refspec.__name__}"]'
                        raise TypeError(err)

                # print(refkind, refattr, list[dbclsname[kind]], 'Rel. backpop', attr[:-2])
                body[refkind][refattr] = sqlmodel.Relationship(back_populates=attr[:-2])
                anno[refkind][refattr] = list[dbclsname[kind]]
            elif hasattr(field.annotation, '__origin__') and field.annotation.__origin__ == dict:  # noqa
                body[kind][attr] = dict_default()
                anno[kind][attr] = field.annotation
            elif hasattr(field.annotation, '__origin__') and field.annotation.__origin__ == list:  # noqa
                if not check_list_T(kind, spec, attr, field.annotation, links, specbyname, models, rmfields):
                    keepattrs[kind].add(attr)
                    body[kind][attr] = list_default()
                    anno[kind][attr] = field.annotation

    # create many to many relations
    for (pair, link), ends in links.items():
        if len(ends) == 1:
            kind, attr, refname = ends[0]
            refcls = specbyname[refname]
            assert not hasattr(refcls, f'{kind}s')
            if link != 'DEFAULT':
                ends.append((refcls.modelkind(), f'{link}{kind}s', models[kind].__name__))
            else:
                ends.append((refcls.modelkind(), f'{kind}s', models[kind].__name__))
        assert len(ends) == 2, f'links of order != 2 not supported, {pair}/{link}, {ends}'
        linkname = f'{str.join("",pair).replace("Spec", "")}{link.title()}Link'
        (kind1, attr1, refname1), (kind2, attr2, refname2) = ends
        assert refname1 in pair and refname2 in pair
        linkbody = {
            f'{attr1}id': sqlmodel.Field(default=None, primary_key=True, foreign_key=f'db{kind1}.id'),
            f'{attr2}id': sqlmodel.Field(default=None, primary_key=True, foreign_key=f'db{kind2}.id'),
        }
        linkanno = {f'{attr1}id': Optional[UUID], f'{attr2}id': Optional[UUID]}
        linkbody['__annotations__'] = linkanno
        if debug:
            print('=================================== link =======================================')
            print(pair, link, ends)
            print(linkbody)
            print(kind1, kind2, flush=True)
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
            rmfields[kind].append(attr)
            # print(f'{kind} {attr} : {anno[kind][attr]} = Relationship({kw})')

    # build the models
    backend_models, trimspecs, remote_props = {}, {}, {}
    for kind, spec in models.items():
        for name, member in spec.__dict__.copy().items():
            if hasattr(member, '__layer__') and member.__layer__ == 'backend':
                body[kind][name] = member.__wrapped__
                delattr(spec, name)
        trimspec = copy_model_cls(spec, f'{spec.__name__}Trim', renamed_attrs=renamed_attrs[kind])
        attrs = set(models[kind].model_fields) - set(body[kind])
        attrs = attrs - {'ispublic', 'ghost', 'datecreated', 'telemetry', 'id', '__annotations__'}
        for attr in rmfields[kind] + list(body[kind]):
            if attr in keepattrs[kind]: continue
            # if attr[0] == '_': continue
            # print(kind, 'remove', attr)
            ano = trimspec.__annotations__.get(attr, None)
            ano = ano or trimspec.model_fields.get(attr, None)
            ano = getattr(ano, '__origin__', None)
            if ano == list or ano == typing.Union and attr[-2:] != 'id':  # noqa
                if attr in trimspec.model_fields: del trimspec.model_fields[attr]
                if hasattr(trimspec, attr): delattr(trimspec, attr)
                if attr in trimspec.__annotations__: del trimspec.__annotations__[attr]
        body[kind]['__annotations__'] = to_sqlalchemy_types(anno[kind])
        if debug:
            print(dbclsname[kind])
            print(body[kind]['__annotations__'])
            for k, v in trimspec.model_fields.items():
                print('      ' if k in body[kind] else '    * ', end='')
                print(f'{k}: {str(v)[:51]}...', flush=True)
            # print('   ', [k for k in trimspec.__dict__.keys() if not k[0] == '_'], flush=True)
        dbcls = type(dbclsname[kind], (backendcls._BackendModelBase, trimspec), body[kind], table=True)
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

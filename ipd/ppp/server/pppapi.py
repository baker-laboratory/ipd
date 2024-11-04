import os
import socket
import sys
from uuid import UUID

import ipd
from ipd import ppp

fastapi = ipd.lazyimport('fastapi', 'fastapi[standard]', pip=True)
pydantic = ipd.lazyimport('pydantic', pip=True)
sqlmodel = ipd.lazyimport('sqlmodel', pip=True)
sqlalchemy = ipd.lazyimport('sqlalchemy', pip=True)
ordset = ipd.lazyimport('ordered_set', pip=True)
yaml = ipd.lazyimport('yaml', 'pyyaml', pip=True)
pymol = ipd.lazyimport('pymol', 'pymol-bundle', mamba=True, channels='-c schrodinger')
uvicorn = ipd.dev.lazyimport('uvicorn', 'uvicorn[standard]', pip=True)
# profile = ipd.dev.timed
profile = lambda f: f

_SERVERMODE = 'ppp' == socket.gethostname()

def servermode():
    return _SERVERMODE

def set_servermode(isserver):
    global _SERVERMODE
    _SERVERMODE = isserver
    # print('_SERVERMODE MODE')

@profile
class PPPBackend(ipd.crud.backend.BackendBase, models=ipd.ppp.spec_models):
    def __init__(self, engine, datadir):
        super().__init__(engine)
        self.datadir = datadir
        set_servermode(True)

    def init_routes(self):
        self.route('/api', self.root, methods=['GET'])
        self.route('/api/create/pollfilecontents', self.create_file_with_content, methods=['POST'])
        self.route('/api/create/pollfiles', self.create_empty_files, methods=['POST'])
        self.route('/api/have/pollfile', self.have_pollfile, methods=['GET'])
        self.route('/api/pollinfo', self.pollinfo, methods=['GET'])

    def initdb(self):
        super().initdb()
        ppp.server.defaults.ensure_init_db(self)

    def root(self) -> None:
        return dict(msg='Hello World')

    def useridmap(self):
        query = 'SELECT id,name FROM dbuser WHERE NOT dbuser.ghost'
        idname = self.session.execute(sqlalchemy.text(query)).fetchall()
        return dict(idname), {name: id for id, name in idname}

    def pollinfo(self, user=None):
        query = ('SELECT TB.id,TB.name AS name,dbuser.name AS user,TB.desc,TB.sym,TB.ligand,'
                 'TB.nchain FROM TB JOIN dbuser ON TB.userid = dbuser.id')
        return self._get_table_info(query, user, 'dbpoll')

    def cmdinfo(self, user=None):
        query = ('SELECT TB.id,TB.name AS name,dbuser.name AS user,TB.desc,TB.sym,TB.ligand,'
                 'TB.minchains,TB.maxchains FROM TB JOIN dbuser ON TB.userid = dbuser.id')
        return self._get_table_info(query, user, 'dbpymolcmd')

    def _get_table_info(self, query, user, table):
        if user == 'None': user = None
        if user: assert self.buser(name=user)
        if user and user != 'admin':
            query += f' WHERE NOT TB.ghost AND (TB.ispublic OR dbuser.name = \'{user}\')'
        query = query.replace('TB', table)
        # print(query)
        result = self.session.execute(sqlalchemy.text(f'{query};')).fetchall()
        result = list(map(list, result))
        return result

    def reviews_fname(self, fname):
        fname = fname.replace('__DIRSEP__', '/')
        files = self.session.exec(sqlmodel.select(DBPollFile).where(DBPollFile.fname == fname))
        rev = ordset.OrderedSet()
        for f in files:
            rev |= f.reviews
        return list(rev)

    def have_pollfile(self, pollid: UUID, fname: str):
        poll = self.poll(dict(id=pollid))
        assert poll, f'invalid pollid {pollid} {self.npolls()}'
        newfname = self.permafname_name(poll, fname)
        return os.path.exists(newfname), newfname

    def permafname_name(self, poll, fname):
        pollname = poll.name.replace(' ', '_').replace('/', '\\')
        path = os.path.join(self.datadir, 'poll', f'{pollname}__{poll.id}', 'reviewed')
        os.makedirs(path, exist_ok=True)
        newfname = os.path.join(path, fname.replace('/', '\\'))
        return newfname

    def create_file_with_content(self, file: 'DBPollFile'):
        assert file.filecontent
        assert file.permafname
        mode = 'wb' if file.permafname.endswith('.bcif') else 'w'
        with open(file.permafname, mode) as out:
            out.write(file.filecontent)

    def create_empty_files(self, files: list[ppp.PollFileSpec]):
        # print('CREATE empty files', len(files))
        for file in files:
            assert not file.filecontent.strip()
            self.session.add(DBPollFile(**file.model_dump()))
        self.session.commit()
        poll = self.poll(dict(id=files[0].pollid))

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# the client models come from here
# class DBPoll
# class DBFileKind
# class DBPollFile
# class DBReview
# class DBReviewStep
# class DBPymolCMD
# class DBWorkflow
# class DBFlowStep
# class DBUser
# class DBGroup
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
for cls in PPPBackend.__backend_models__.values():
    globals()[cls.__name__] = cls

def pymol_launch():
    stdout = sys.stdout
    stderr = sys.stderr
    pymol.finish_launching(['pymol', '-xiqckK'])
    sys.stdout = stdout
    sys.stderr = stderr
    pymol.cmd.set('suspend_updates', 'on')

def run(**kw):
    return ipd.crud.run[PPPBackend, ipd.ppp.PPPClient](**kw)

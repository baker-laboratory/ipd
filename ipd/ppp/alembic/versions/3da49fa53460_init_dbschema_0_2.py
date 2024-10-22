"""init dbschema 0.2

Revision ID: 3da49fa53460
Revises: 
Create Date: 2024-10-18 00:35:18.213326

"""
from typing import Sequence, Union

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '3da49fa53460'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('dbfilekind', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('filekind', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.PrimaryKeyConstraint('id'))
    op.create_table('dbuser', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('fullname', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('name'))
    op.create_table('dbgroup', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('userid', sa.Uuid(), nullable=False),
                    sa.Column('ispublic', sa.Boolean(), nullable=False),
                    sa.Column('telemetry', sa.Boolean(), nullable=False),
                    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.ForeignKeyConstraint(
                        ['userid'],
                        ['dbuser.id'],
                    ), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('name'))
    op.create_table('dbpymolcmd', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('userid', sa.Uuid(), nullable=False),
                    sa.Column('ispublic', sa.Boolean(), nullable=False),
                    sa.Column('telemetry', sa.Boolean(), nullable=False),
                    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('desc', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('cmdon', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('cmdoff', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('cmdstart', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('onstart', sa.Boolean(), nullable=False),
                    sa.Column('ligand', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('sym', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('minchains', sa.Integer(), nullable=False),
                    sa.Column('maxchains', sa.Integer(), nullable=False),
                    sa.Column('cmdcheck', sa.Boolean(), nullable=False),
                    sa.ForeignKeyConstraint(
                        ['userid'],
                        ['dbuser.id'],
                    ), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('name'))
    op.create_table('dbworkflow', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('ispublic', sa.Boolean(), nullable=False),
                    sa.Column('telemetry', sa.Boolean(), nullable=False),
                    sa.Column('desc', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('ordering', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('userid', sa.Uuid(), nullable=False),
                    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.ForeignKeyConstraint(
                        ['userid'],
                        ['dbuser.id'],
                    ), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('name'))
    op.create_table('useruserdefaultlink', sa.Column('followersid', sa.Uuid(), nullable=False),
                    sa.Column('followingid', sa.Uuid(), nullable=False),
                    sa.ForeignKeyConstraint(
                        ['followersid'],
                        ['dbuser.id'],
                    ), sa.ForeignKeyConstraint(
                        ['followingid'],
                        ['dbuser.id'],
                    ), sa.PrimaryKeyConstraint('followersid', 'followingid'))
    op.create_table('dbflowstep', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('workflowid', sa.Uuid(), nullable=False),
                    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('index', sa.Integer(), nullable=False),
                    sa.Column('taskgen', sa.JSON(), nullable=True),
                    sa.Column('instructions', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.ForeignKeyConstraint(
                        ['workflowid'],
                        ['dbworkflow.id'],
                    ), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('name'))
    op.create_table('dbpoll', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('ispublic', sa.Boolean(), nullable=False),
                    sa.Column('telemetry', sa.Boolean(), nullable=False),
                    sa.Column('desc', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('path', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('cmdstart', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('cmdstop', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('sym', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('nchain', sa.Integer(), nullable=False),
                    sa.Column('ligand', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('userid', sa.Uuid(), nullable=False),
                    sa.Column('name', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('workflowid', sa.Uuid(), nullable=True),
                    sa.Column('enddate', sa.DateTime(), nullable=False),
                    sa.ForeignKeyConstraint(
                        ['userid'],
                        ['dbuser.id'],
                    ), sa.ForeignKeyConstraint(
                        ['workflowid'],
                        ['dbworkflow.id'],
                    ), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('name'))
    op.create_table('groupuserdefaultlink', sa.Column('groupsid', sa.Uuid(), nullable=False),
                    sa.Column('usersid', sa.Uuid(), nullable=False),
                    sa.ForeignKeyConstraint(
                        ['groupsid'],
                        ['dbuser.id'],
                    ), sa.ForeignKeyConstraint(
                        ['usersid'],
                        ['dbgroup.id'],
                    ), sa.PrimaryKeyConstraint('groupsid', 'usersid'))
    op.create_table('dbpollfile', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('fname', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('tag', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('permafname', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('filecontent', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('pollid', sa.Uuid(), nullable=False),
                    sa.Column('parentid', sa.Uuid(), nullable=True),
                    sa.Column('filekindid', sa.Uuid(), nullable=True),
                    sa.ForeignKeyConstraint(
                        ['filekindid'],
                        ['dbfilekind.id'],
                    ), sa.ForeignKeyConstraint(
                        ['parentid'],
                        ['dbpollfile.id'],
                    ), sa.ForeignKeyConstraint(
                        ['pollid'],
                        ['dbpoll.id'],
                    ), sa.PrimaryKeyConstraint('id'))
    op.create_table('dbreview', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('ispublic', sa.Boolean(), nullable=False),
                    sa.Column('telemetry', sa.Boolean(), nullable=False),
                    sa.Column('grade', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('comment', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('userid', sa.Uuid(), nullable=False),
                    sa.Column('pollid', sa.Uuid(), nullable=False),
                    sa.Column('pollfileid', sa.Uuid(), nullable=False),
                    sa.Column('workflowid', sa.Uuid(), nullable=False),
                    sa.Column('durationsec', sa.Integer(), nullable=False),
                    sa.ForeignKeyConstraint(
                        ['pollfileid'],
                        ['dbpollfile.id'],
                    ), sa.ForeignKeyConstraint(
                        ['pollid'],
                        ['dbpoll.id'],
                    ), sa.ForeignKeyConstraint(
                        ['userid'],
                        ['dbuser.id'],
                    ), sa.ForeignKeyConstraint(
                        ['workflowid'],
                        ['dbworkflow.id'],
                    ), sa.PrimaryKeyConstraint('id'))
    op.create_table('dbreviewstep', sa.Column('id', sa.Uuid(), nullable=False),
                    sa.Column('ghost', sa.Boolean(), nullable=False),
                    sa.Column('datecreated', sa.DateTime(), nullable=False),
                    sa.Column('props', sa.JSON(), nullable=True), sa.Column('attrs', sa.JSON(), nullable=True),
                    sa.Column('reviewid', sa.Uuid(), nullable=False),
                    sa.Column('flowstepid', sa.Uuid(), nullable=False),
                    sa.Column('task', sa.JSON(), nullable=True),
                    sa.Column('grade', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('comment', sqlmodel.sql.sqltypes.AutoString(), nullable=False),
                    sa.Column('durationsec', sa.Integer(), nullable=False),
                    sa.ForeignKeyConstraint(
                        ['flowstepid'],
                        ['dbflowstep.id'],
                    ), sa.ForeignKeyConstraint(
                        ['reviewid'],
                        ['dbreview.id'],
                    ), sa.PrimaryKeyConstraint('id'))
    # ### end Alembic commands ###

def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('dbreviewstep')
    op.drop_table('dbreview')
    op.drop_table('dbpollfile')
    op.drop_table('groupuserdefaultlink')
    op.drop_table('dbpoll')
    op.drop_table('dbflowstep')
    op.drop_table('useruserdefaultlink')
    op.drop_table('dbworkflow')
    op.drop_table('dbpymolcmd')
    op.drop_table('dbgroup')
    op.drop_table('dbuser')
    op.drop_table('dbfilekind')
    # ### end Alembic commands ###

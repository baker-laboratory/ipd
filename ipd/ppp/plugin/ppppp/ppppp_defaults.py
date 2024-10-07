import getpass

state_defaults = dict(
    ispublic=True,
    reviewed=set(),
    prefetch=3,
    review_action='cp $file $pppdir/$poll/$grade_$filebase',
    do_review_action=False,
    findcmd='',
    findpoll='',
    shuffle=False,
    use_rsync=False,
    hide_invalid=True,
    showallcmds=False,
    pymol_sc_repr='sticks',
    active_cmds=set(),
    activepoll=None,
    activepollindex=0,
    files=set(),
    serveraddr='ppp.ipd',
    user=getpass.getuser(),
)
state_types = dict(
    cmds='global',
    activepoll='global',
    polls='global',
    active_cmds='perpoll',
    reviewed='perpoll',
    pymol_view='perpoll',
    serveraddr='global',
    user='global',
)

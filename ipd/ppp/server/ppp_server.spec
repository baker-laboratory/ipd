Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
    mkdir -p $APPTAINER_ROOTFS/prettier_protein_project/lib

%files
    /home/sheffler/sw/Miniforge3-Linux-x86_64.sh /opt

%post
    apt update && apt install -y git
    bash /opt/Miniforge3-Linux-x86_64.sh -b -p /opt/mamba
    /opt/mamba/bin/mamba install -c schrodinger pymol-bundle sqlmodel fastapi uvicorn ordered-set pyyaml "pip<25" icecream RestrictedPython psycopg2 mysql-connector-python
    ls /prettier_protein_project/lib/ipd
    /opt/mamba/bin/pip install -e /prettier_protein_project/lib/ipd
    /opt/mamba/bin/pip install -e /prettier_protein_project/lib/willutil
    /opt/mamba/bin/pip install -e /prettier_protein_project/lib/wills_pymol_crap

%runscript
    echo updating library ipd from github branch ppp
    git config --global --add safe.directory /prettier_protein_project/lib/ipd
    echo updating library willutil from github
    git config --global --add safe.directory /prettier_protein_project/lib/willutil
    echo updating library wills_pymol_crap
    git config --global --add safe.directory /prettier_protein_project/lib/wills_pymol_crap
    cd /prettier_protein_project/lib/ipd && git pull
    cd /prettier_protein_project/lib/willutil && git pull
    cd /prettier_protein_project/lib/wills_pymol_crap && git pull
    cd /prettier_protein_project/server
    /opt/mamba/bin/python -m ipd.ppp.server "$@"

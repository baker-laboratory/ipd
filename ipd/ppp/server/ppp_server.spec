Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
    mkdir -p $APPTAINER_ROOTFS/prettier_protein_project/lib
    touch /etc/localtime

%files
    /home/sheffler/sw/Miniforge3-Linux-x86_64.sh /opt

%post
    apt update && apt install -y git
    bash /opt/Miniforge3-Linux-x86_64.sh -b -p /opt/mamba
    /opt/mamba/bin/mamba install -c schrodinger pymol-bundle sqlmodel fastapi uvicorn ordered-set pyyaml "pip<25" icecream RestrictedPython psycopg2 mysql-connector-python assertpy
    /opt/mamba/bin/pip install -e /prettier_protein_project/lib/ipd
    /opt/mamba/bin/pip install -e /prettier_protein_project/lib/willutil
    /opt/mamba/bin/pip install -e /prettier_protein_project/lib/wills_pymol_crap
    git config --global --add safe.directory /prettier_protein_project/lib/ipd
    git config --global --add safe.directory /prettier_protein_project/lib/willutil
    git config --global --add safe.directory /prettier_protein_project/lib/wills_pymol_crap

%runscript
    echo updating library ipd from github branch prettier_protein_project
    cd /prettier_protein_project/lib/ipd && git pull
    echo updating library willutil from github
    cd /prettier_protein_project/lib/willutil && git pull
    echo updating library wills_pymol_crap
    cd /prettier_protein_project/lib/wills_pymol_crap && git pull
    cd /prettier_protein_project/server
    /opt/mamba/bin/python -m ipd.prettier_protein_project.server --datadir /prettier_protein_project/server/data "$@"

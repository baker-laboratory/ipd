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
    git config --global --add safe.directory /prettier_protein_project/lib/ipd
    git config --global --add safe.directory /prettier_protein_project/lib/willutil
    git config --global --add safe.directory /prettier_protein_project/lib/wills_pymol_crap
    ls /prettier_protein_project/lib/ipd
    /opt/mamba/bin/pip install -e /prettier_protein_project/lib/ipd
    /opt/mamba/bin/pip install -e /prettier_protein_project/lib/willutil
    /opt/mamba/bin/pip install -e /prettier_protein_project/lib/wills_pymol_crap

%runscript
    /opt/mamba/bin/python -m ipd.ppp.server --update_libs_in /prettier_protein_project/lib "$@"

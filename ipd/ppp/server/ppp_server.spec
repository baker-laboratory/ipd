Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
    mkdir -p $APPTAINER_ROOTFS/ppp
    touch $APPTAINER_ROOTFS/etc/localtime
    touch $APPTAINER_ROOTFS/etc/hosts
    touch $APPTAINER_ROOTFS/root/.pymolrc

%files
    /home/sheffler/sw/Miniforge3-Linux-x86_64.sh /opt

%post
    mkdir -p /ppp/lib /ppp/data && rm -rf /ppp/lib/*
    apt update && apt install -q -y git libglib2.0-0t64
    git clone -b ppp https://github.com/baker-laboratory/ipd /ppp/lib/ipd
    git clone https://github.com/willsheffler/willutil /ppp/lib/willutil
    git clone https://github.com/willsheffler/wills_pymol_crap /ppp/lib/wills_pymol_crap
    bash /opt/Miniforge3-Linux-x86_64.sh -b -p /opt/mamba
    /opt/mamba/bin/mamba install -q -y -c schrodinger pymol-bundle sqlmodel fastapi[standard] uvicorn[standard] ordered-set pyyaml "pip<25" icecream RestrictedPython psycopg2 mysql-connector-python assertpy uvloop alembic alembic-utils
    /opt/mamba/bin/pip install alembic-postgresql-enum
    for lib in ipd willutil wills_pymol_crap; do
        /opt/mamba/bin/pip install -e /ppp/lib/$lib
        git config --global --add safe.directory /ppp/lib/$lib
    done

%runscript
    for lib in ipd willutil wills_pymol_crap; do
        echo updating library $lib
        cd /ppp/lib/$lib && git pull
    done
    cd /ppp && /opt/mamba/bin/python -m ipd.ppp.server --datadir /ppp/data "$@"

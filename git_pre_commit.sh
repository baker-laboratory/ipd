#!/bin/bash

gitroot=$(git rev-parse --show-toplevel)
echo git root dir found: $gitroot
cd $gitroot

if [ $(basename $gitroot) == 'ipd' ]; then
    ipd="$gitroot"
    src=ipd
elif [ $(basename $gitroot) == 'rf2aa' ]; then
    ipd="$gitroot/lib/ipd"
    src=rf2aa
else
    ipd="$gitroot/lib/ipd"
    src='rf_diffusion'
fi

validate-pyproject pyproject.toml

echo running: ruff check --fix $src
if ruff check --fix $src; then
    echo ruff ok;
else
    echo ruff failed; exit 1;
fi

if [ -f .pyright_hash_last_commit ]; then
    cmd="PYTHONPATH=$ipd python -m ipd code pyright $src --hashfile '.pyright_hash_last_commit'"
    echo $cmd
    eval $cmd
    if [ $? == 0 ]; then
        echo pyright pass
    else
        echo pyright fail
        exit 1
    fi
    git add .pyright_hash_last_commit
fi

cmd="PYTHONPATH=$ipd python -m ipd code yapf $src"
echo $cmd
eval $cmd
if [ $? == 0 ]; then
    echo files all formatted
else
    echo yapf formatted some files, retry commit
    exit 1
fi

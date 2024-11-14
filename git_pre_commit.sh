#!/bin/bash

gitroot=$(git rev-parse --show-toplevel)
echo git root dir found: $gitroot

if [ $(basename $gitroot) == 'ipd' ]; then
    ipd="$gitroot"
    src=ipd
elif [ $(basename $gitroot) == 'rf2aa' ]; then
    ipd="$gitroot/lib/ipd"
    src=rf2aa
else
    ipd="$gitroot/lib/rf2aa/lib/ipd"
    src='rf_diffusion'
fi

echo running: ruff check --fix $src
if ruff check --fix $src; then
    echo ruff ok;
else
    echo ruff failed; exit 1;
fi

cmd="PYTHONPATH=$ipd python $ipd/ipd/tools/yapf_fast.py $src"
echo $cmd
eval $cmd

if [ $? == 0 ]; then
    echo files all formatted
else
    echo some files changed, retry commit
    exit 1
fi


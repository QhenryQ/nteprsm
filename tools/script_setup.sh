#!/bin/bash

CWD=$(pwd)
cd $(dirname $0)/..
REPO_ROOT=$(pwd)
cd $CWD

TOOLS_DIR=$REPO_ROOT/tools
PYTHON_VER=3.12.1
VIRTUAL_PY=venv.ntep-rsm

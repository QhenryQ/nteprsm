#!/bin/bash

# Store the current working directory
CWD=$(pwd)

# Navigate to the repository root
cd $(dirname $0)/..
REPO_ROOT=$(pwd)
cd $CWD

# Define key variables
TOOLS_DIR=$REPO_ROOT/tools
VIRTUAL_PY=venv.nteprsm

# Prompt user to set up a data directory
DEFAULT_DATA_DIR="$REPO_ROOT/data"
echo "Please specify the data directory to the shared Google Drive Folder(default: $DEFAULT_DATA_DIR):"
read -r DATA_DIR
DATA_DIR=${DATA_DIR:-$DEFAULT_DATA_DIR}
echo "Using data directory: $DATA_DIR"

# Update settings.py with the new DATA_DIR as a Path object from pathlib
SETTINGS_FILE="$REPO_ROOT/settings.py"
if [ -f "$SETTINGS_FILE" ]; then
    echo "UPDATING DATA_DIR IN settings.py"
    sed -i.bak "s|DATA_DIR = .*|DATA_DIR = Path(\"$DATA_DIR\")|" "$SETTINGS_FILE"
    echo "DATA_DIR updated to: Path(\"$DATA_DIR\")"
else
    echo "ERROR: settings.py not found at $SETTINGS_FILE"
    exit 1
fi

# Ensure Python 3.12 is installed
echo "CHECKING PYTHON INSTALLATION"

PYTHON_VERSION_REQUIRED="3.12"
if ! [ -x "$(command -v python3)" ]; then
    echo "PYTHON NOT FOUND. INSTALLING PYTHON $PYTHON_VERSION_REQUIRED"
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        brew install python@$PYTHON_VERSION_REQUIRED
    elif [ "$(uname)" == "Linux" ]; then
        # Linux
        sudo apt update
        sudo apt install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt update
        sudo apt install -y python$PYTHON_VERSION_REQUIRED python3-pip
    elif [[ "$(uname -s)" == *"MINGW"* || "$(uname -s)" == *"CYGWIN"* ]]; then
        # Windows (Git Bash or Cygwin)
        echo "Please install Python $PYTHON_VERSION_REQUIRED manually from https://www.python.org/downloads/"
        exit 1
    else
        echo "Unsupported OS. Please install Python $PYTHON_VERSION_REQUIRED manually."
        exit 1
    fi
fi

CURRENT_PYTHON_VERSION=$(python3 --version | awk '{print $2}')
if [[ "$CURRENT_PYTHON_VERSION" != "$PYTHON_VERSION_REQUIRED"* ]]; then
    if [[ "$CURRENT_PYTHON_VERSION" =~ ^3\.12\.* ]]; then
        echo "PYTHON $CURRENT_PYTHON_VERSION IS COMPATIBLE WITH REQUIRED VERSION $PYTHON_VERSION_REQUIRED."
    else
        echo "CURRENT PYTHON VERSION ($CURRENT_PYTHON_VERSION) IS NOT COMPATIBLE WITH $PYTHON_VERSION_REQUIRED. PLEASE INSTALL PYTHON $PYTHON_VERSION_REQUIRED."
        exit 1
    fi
else
    echo "PYTHON $CURRENT_PYTHON_VERSION IS INSTALLED."
fi

# Check Poetry installation
echo "CHECKING POETRY INSTALLATION"

if ! [ -x "$(command -v poetry)" ]; then
    echo "INSTALLING POETRY VERSION 2.1.0"
    curl -sSL https://install.python-poetry.org | python3 - --version 2.1.0
else
    export MINIMUM_POETRY_VERSION=2.0.0
    export MAXIMUM_POETRY_VERSION=2.2.0
    export CURRENT_POETRY_VERSION="$(poetry -V | grep -Eo '([0-9]\.[0-9]\.[0-9])+')"
    function ver { printf "%03d%03d%03d" $(echo "$1" | tr '.' ' '); }

    if [ $(ver $CURRENT_POETRY_VERSION) -lt $(ver $MINIMUM_POETRY_VERSION) ]; then
        echo "UPDATING POETRY VERSION v$CURRENT_POETRY_VERSION < v$MINIMUM_POETRY_VERSION"

        echo "UNINSTALLING CURRENT POETRY INSTALLATION"
        curl -sSL https://install.python-poetry.org | python3 - --uninstall

        echo "INSTALLING POETRY VERSION 2.1.0"
        curl -sSL https://install.python-poetry.org | python3 - --version 2.1.0

    elif [ $(ver $CURRENT_POETRY_VERSION) -gt $(ver $MAXIMUM_POETRY_VERSION) ]; then

        echo "DOWNGRADING POETRY VERSION v$CURRENT_POETRY_VERSION > v$MAXIMUM_POETRY_VERSION"

        echo "UNINSTALLING CURRENT POETRY INSTALLATION"
        curl -sSL https://install.python-poetry.org | python3 - --uninstall

        echo "INSTALLING POETRY VERSION 2.1.0"
        curl -sSL https://install.python-poetry.org | python3 - --version 2.1.0

    else
        echo "COOL. YOU HAD THE RIGHT POETRY VERSION"
    fi
fi

# Poetry requires its bin directory in your `PATH`
if ! [[ ":$PATH:" == *":$HOME/.local/bin:"* ]]; then
    echo "ADDING POETRY'S BIN TO YOUR PATH."
    export PATH="$HOME/.local/bin:$PATH"
fi

# Add poetry-plugin-shell to enable `poetry shell`
echo "ADDING poetry-plugin-shell TO POETRY"
poetry self add poetry-plugin-shell

# Install dependencies
echo "INSTALLING DEPENDENCIES"

poetry install --all-extras
poetry shell

echo "ALL DONE!"

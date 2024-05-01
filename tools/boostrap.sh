#!/bin/bash
. $(dirname $0)/script_setup.sh

# check poetry installation
echo "CHECKING POETRY INSTALLATION"

if ! [ -x "$(command -v poetry)" ]; then
  echo "INSTALLING POETRY VERSION 1.7.1"
  curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1
else
  export MINIMUM_POETRY_VERSION=1.7.1
  export MAXIMUM_POETRY_VERSION=1.7.1
  export CURRENT_POETRY_VERSION="$(poetry -V | grep -Eo '([0-9]\.[0-9]\.[0-9])+')"
  function ver { printf "%03d%03d%03d" $(echo "$1" | tr '.' ' '); }

  if [ $(ver $CURRENT_POETRY_VERSION) -lt $(ver $MINIMUM_POETRY_VERSION) ]; then
    echo "UPDATING POETRY VERSION v$CURRENT_POETRY_VERSION < v$MINIMUM_POETRY_VERSION"

    echo "UNINSTALLING CURRENT POETRY INSTALLATION"
    curl -sSL https://install.python-poetry.org | python3 - --uninstall

    echo "INSTALLING POETRY VERSION 1.7.1"
    curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1

  elif [ $(ver $CURRENT_POETRY_VERSION) -gt $(ver $MAXIMUM_POETRY_VERSION) ]; then

    echo "DOWNGRADING POETRY VERSION v$CURRENT_POETRY_VERSION > v$MAXIMUM_POETRY_VERSION"

    echo "UNINSTALLING CURRENT POETRY INSTALLATION"
    curl -sSL https://install.python-poetry.org | python3 - --uninstall

    echo "INSTALLING POETRY VERSION 1.5.1"
    curl -sSL https://install.python-poetry.org | python3 - --version 1.7.1

  else
    echo "COOL. YOU HAD THE RIGHT POETRY VERSION"
  fi
fi

# Poetry requires its bin directory in your `PATH`
if ! [[ ":$PATH:" == *":$HOME/.local/bin:"* ]]; then
  echo "ADDING POETRY'S BIN TO YOUR PATH."
  export PATH="$HOME/.local/bin:$PATH"
fi

# install dependencies
echo "INSTALL DEPENDENCIES"

poetry env use 3.12
poetry install --all-extras
poetry shell

echo "ALL DONE!"

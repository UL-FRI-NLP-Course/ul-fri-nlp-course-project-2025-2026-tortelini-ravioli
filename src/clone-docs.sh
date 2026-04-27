#!/bin/bash

if ! command -v git &> /dev/null
then
    echo "git could not be found. Please install git to run this script."
    exit
fi

git clone --no-checkout https://github.com/grafana/grafana.git
cd grafana
git sparse-checkout init
git sparse-checkout set docs/sources

git checkout main
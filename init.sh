#!/bin/bash

python3 -m venv venv
venv/bin/pip3 install -r requirements.txt
mkdir -pv img
ln --interactive --relative --symbolic --verbose $PWD/img static/img

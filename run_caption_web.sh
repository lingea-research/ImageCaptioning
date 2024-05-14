#!/usr/bin/env bash
port=8080
model_name=vitgpt

if [ -n "$1" ]; then
  if [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "using port ${1} for gunicorn"
    port=$1
  else
    echo "invalid port '${1}', quitting"
    echo "usage: $(basename $0) <gunicorn port number>"
    exit
fi; fi

source ./venv/bin/activate \
&& gunicorn --reload \
  --threads 1 \
  --worker-connections 1 \
  --workers 1 \
  --timeout 300 \
  -b localhost:$port \
  caption_server:app \
&& deactivate


#!/usr/bin/env bash
models='vitgpt git_base blip_base git_large blip_large'
test_script=caption.py

usage() {
  echo "usage: $(basename $0): <images_dir>"
  exit
}
[ -z "$1" -o ! -d "$1" ] && usage

images=`find "${1}" \( -iname '*.png' -o -iname '*.jp*g' \) \
  | ./venv/bin/python3 -c "import utilities as ut; ut.urlencode_stdin()" \
  | paste -sd' '`

for m in $models; do
  ./venv/bin/python3 $test_script $m $images
done


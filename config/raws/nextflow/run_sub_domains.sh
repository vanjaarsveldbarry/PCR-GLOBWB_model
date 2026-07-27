#!/bin/bash
# Launch the RAWS sub-basin run.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

set +u
eval "$(pixi shell-hook --manifest-path "${repo_root}/pixi.toml")"
set -u

nextflow run "${script_dir}/main.nf" "$@"

wait

#TODO: the output files are all still seperate clones, how can I make it write
# to a single output zarr. 
#What will you do if the RAM required exceeds 1 node? 
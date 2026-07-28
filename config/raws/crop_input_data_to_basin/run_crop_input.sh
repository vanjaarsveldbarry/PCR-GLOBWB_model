set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../../.." && pwd -P)"

INI="$repo_root/config/raws/5min.ini"
CLONE_ID=182
CLONE="/scratch/depfg/7006713/RAWS/sub_domains/output/clone_maps/M${CLONE_ID}.clone.map"
OUTPUT="/scratch/depfg/7006713/RAWS/pcrglob_model/input/M${CLONE_ID}"

set +u
eval "$(pixi shell-hook --manifest-path "$repo_root/pixi.toml")"
set -u

taskset -c 0-95 python "$script_dir/crop_input_to_clone.py" \
    --ini "$INI" \
    --clone "$CLONE" \
    --output "$OUTPUT"


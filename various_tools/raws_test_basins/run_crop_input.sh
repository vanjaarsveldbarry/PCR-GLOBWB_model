set -euo pipefail

# --- locations -------------------------------------------------------------
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd -P)"
repo_root="$(cd -- "$script_dir/../.." && pwd -P)"   # PCR-GLOBWB_model/ (holds pixi.toml)

# --- inputs (edit these) ---------------------------------------------------
INI="$repo_root/config/raws/5min.ini"
CLONE="/scratch/depfg/7006713/RAWS/sub_domains/output/clone_maps/M104.clone.map"
OUTPUT="/scratch/depfg/7006713/RAWS/pcrglob_model/input"
BUFFER_CELLS=5
JOBS=8                       # files cropped in parallel

# --- run -------------------------------------------------------------------
pixi run --manifest-path "$repo_root/pixi.toml" \
    taskset -c 0-95 python "$script_dir/crop_input_to_clone.py" \
        --ini "$INI" \
        --clone "$CLONE" \
        --output "$OUTPUT" \
        --buffer-cells "$BUFFER_CELLS" \
        --jobs "$JOBS" \
        "$@"

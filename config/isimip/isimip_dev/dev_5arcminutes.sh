#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$CONFIG_DIR/../../.." && pwd)"

# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
DATA_CATALOGUE_DIR=/scratch/depfg/7006713/isimip_data_catalogue/isimip_parameterization/data_catalogue/CATALOGUE/TEST_CASES/
DATA_DIR="${REPO_ROOT}/../data"

START_DATE=2000-01-01
END_DATE=2000-01-07

INPUT_DIR="$DATA_DIR/CATALOGUE/TEST_CASES/rhine"
CLONE_MAP="clone_maps/rhine_5arcminutes"

# ---------------------------------------------------------------------------
# get data
# ---------------------------------------------------------------------------
mkdir -p "$DATA_DIR/CATALOGUE"

pixi run --manifest-path "$REPO_ROOT/pixi.toml" \
    rsync -avh --delete --partial --info=progress2 \
    "eejit:${DATA_CATALOGUE_DIR%/}" "$DATA_DIR/CATALOGUE"

# ---------------------------------------------------------------------------
# 1/2 PCR-GLOBWB on its own
# ---------------------------------------------------------------------------
pixi run --manifest-path "$REPO_ROOT/pixi.toml" \
    python "$REPO_ROOT/model/deterministic_runner_with_arguments.py" \
    "$REPO_ROOT/config/isimip/run_with_args/pcrglobwb_5arcminutes.ini" \
    -mod "$DATA_DIR/output/5arcminutes/pcrglobwb" \
    -mid "$INPUT_DIR" \
    -clonemap "$CLONE_MAP" \
    -sd "$START_DATE" \
    -ed "$END_DATE"
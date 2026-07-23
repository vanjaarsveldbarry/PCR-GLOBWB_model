#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 20:00:00
#SBATCH -o /eejit/home/7006713/projects/globalRun0723/5min/oldDownscale/slurmOut/basin%a.out

cd "/eejit/home/7006713/projects/RAWS/pcrglob_model/PCR-GLOBWB_model/model"

# unset PCRASTER_NR_WORKER_THREADS
export PCRASTER_NR_WORKER_THREADS=32

INI_FILE=/scratch/depfg/7006713/RAWS/pcrglob_model/input/5min_cropped.ini
STARTING_DATE=1979-01-01
END_DATE=1979-01-31
_TAG="base"
MAIN_OUTPUT_DIR=/scratch/depfg/7006713/RAWS/pcrglob_model/run_${_TAG}

pixi run --manifest-path "../pixi.toml" python deterministic_runner_with_arguments.py ${INI_FILE} -mod ${MAIN_OUTPUT_DIR} -sd ${STARTING_DATE} -ed ${END_DATE}
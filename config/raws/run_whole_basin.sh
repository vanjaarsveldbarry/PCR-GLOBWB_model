#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 20:00:00
#SBATCH -J missi_test
#SBATCH -o /eejit/home/7006713/projects/RAWS/pcrglob_model/PCR-GLOBWB_model/config/raws/missi_test.out

cd "/eejit/home/7006713/projects/RAWS/pcrglob_model/PCR-GLOBWB_model/model"

# unset PCRASTER_NR_WORKER_THREADS
export PCRASTER_NR_WORKER_THREADS=32

INI_FILE=/scratch/depfg/7006713/RAWS/pcrglob_model/input_M104/5min_cropped.ini
_TAG="base_M104"
MAIN_OUTPUT_DIR=/scratch/depfg/7006713/RAWS/pcrglob_model/output_${_TAG}

eval "$(pixi shell-hook --manifest-path "../pixi.toml")"

python deterministic_runner_with_arguments.py ${INI_FILE} -mod ${MAIN_OUTPUT_DIR}
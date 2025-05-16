#!/bin/bash -l
#SBATCH --job-name=quality
#SBATCH --nodes=1
#SBATCH --cores=10
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --partition=msilarge
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oboiko@umn.edu
#SBATCH --output=model_runs/%A_%a.out
locs_traits=('in1/quality' 'mi1/quality') #'mn1/quality' 'nc1/quality' 'nj2/quality' 'ok1/quality' 'ut1/quality')
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID"
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
loc_trait=${locs_traits[$SLURM_ARRAY_TASK_ID]}
data_path=kb2017/${loc_trait}.csv
output_string=$(echo $loc_trait | tr '/' '_')
model_output_file=model_seasonality_${output_string}.pkl
source activate nteprsm_env
python nteprsm/model_seasonality.py config/nteprsm_config_seasonality.yml $data_path  model_runs/seasonality_$output_string $model_output_file
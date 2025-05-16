#!/bin/bash -l
#SBATCH --job-name=old
#SBATCH --nodes=1
#SBATCH --cores=10
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --array=0-0
#SBATCH --partition=msilarge
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oboiko@umn.edu
#SBATCH --output=model_runs/%A_%a.out
locs_traits=('nj2/texture') #( 'nj2/color' 'nj2/density' 'nj2/drought_quality' 'nj2/rust' 'nj2/spring_greenup' 'nj2/texture' 'nj2/uniformity')
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID"
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
loc_trait=${locs_traits[$SLURM_ARRAY_TASK_ID]}
data_path=kb2017/${loc_trait}.csv
output_string=$(echo $loc_trait | tr '/' '_')
model_output_file=model_old_${output_string}.pkl
source activate nteprsm_env
python nteprsm/model_old.py config/nteprsm_config_old.yml $data_path model_runs/old_$output_string $model_output_file
#!/bin/bash -l
#SBATCH --job-name=spatial
#SBATCH --nodes=1
#SBATCH --cores=10
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --array=0-44
#SBATCH --partition=msilarge
#SBATCH --mail-type=ALL
#SBATCH --mail-user=oboiko@umn.edu
#SBATCH --output=model_runs/%A_%a.out
locs_traits=('nj2/color' 'nj2/density' 'nj2/drought_quality' 'nj2/rust' 'nj2/spring_greenup' 'nj2/texture' 'nj2/uniformity' 'in1/color' 'in1/density' 'in1/dollar_spot' 'in1/rust' 'in1/seedhead' 'in1/seedling_vigor' 'in1/spring_greenup' 'in1/texture' 'in1/winter_color' 'mi1/color' 'mi1/density' 'mi1/rust' 'mi1/texture' 'mi1/uniformity' 'mn1/color' 'mn1/density' 'mn1/pink_snow_mold' 'mn1/seedling_vigor' 'mn1/spring_greenup' 'mn1/texture' 'nc1/color' 'nc1/density' 'nc1/fall_color' 'nc1/seedhead' 'nc1/seedling_vigor' 'nc1/spring_greenup' 'nc1/summer_patch' 'nc1/texture' 'nc1/uniformity'  'ok1/color' 'ok1/density' 'ok1/rust' 'ok1/spring_greenup' 'ok1/texture' 'ut1/color' 'ut1/density' 'ut1/spring_greenup' 'ut1/texture')
echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID"
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
loc_trait=${locs_traits[$SLURM_ARRAY_TASK_ID]}
data_path=kb2017/${loc_trait}.csv
output_string=$(echo $loc_trait | tr '/' '_')
model_output_file=fit_spatial_${output_string}.pkl
source activate nteprsm_env
python nteprsm/model_umn.py config/spatial_model.yml $data_path model_runs/spatial_$output_string $model_output_file
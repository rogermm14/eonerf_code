# run EO-NeRF by Roger Marí
# full paper: "Multi-Date Earth Observation NeRF: The Detail Is in the Shadows" (CVPR Workshops 2023)
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# hyperparams and key vars
aoi_id=$1
suffix=$2
gpu_id=1
downsample_factor=2
n_samples=128
n_importance=0
fc_units=256
training_iters=300000
batch_size=1024

# set paths
errs="$aoi_id"_errors.txt
root_dir="/mnt/adisk/roger/Datasets/SatNeRF/root_dir/crops_rpcs_ba_v2/$aoi_id"
cache_dir="/mnt/adisk/roger/Datasets/SatNeRF/cache_dir_utm/crops_rpcs_ba_v2/"$aoi_id"_ds"$downsample_factor
img_dir="/mnt/adisk/roger/Datasets/DFC2019/Track3-RGB-crops/$aoi_id"
out_dir="/mnt/adisk/roger/eonerfacc_logs_latest"
gt_dir="/mnt/adisk/roger/Datasets/DFC2019/Track3-Truth"
shadow_masks_dir="/mnt/adisk/roger/Datasets/DFC2019/Shadows-pred_v2/Track3-RGB-crops/"$aoi_id
logs_dir=$out_dir/logs
ckpts_dir=$out_dir/ckpts
errs_dir=$out_dir/errs
mkdir -p $errs_dir

# run model
model="eo-nerf"
exp_name="$timestamp"_"$aoi_id"_eonerfacc_7views_shadowsupervision
custom_args="--exp_name "$exp_name" --model $model --img_downscale $downsample_factor --max_train_steps $training_iters --fc_units $fc_units --n_samples $n_samples --n_importance $n_importance --geometric_shadows --radiometric_normalization --batch_size $batch_size" 
extra_args="$custom_args --subset_Nviews 7 --shadow_masks_dir $shadow_masks_dir"
errs=$errs_dir/"$exp_name"_errors.txt
echo -n "" > $errs
python3 train_eonerf.py --root_dir $root_dir --img_dir $img_dir --cache_dir $cache_dir  --ckpts_dir $ckpts_dir --logs_dir $logs_dir --gt_dir $gt_dir --gpu_id $gpu_id $extra_args #2>> $errs


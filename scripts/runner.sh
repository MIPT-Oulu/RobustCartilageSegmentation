#!/bin/bash

# -------------------------------- Prepare datasets ------------------------------------
(cd ../rocaseg/datasets &&
 echo "prepare_dataset" &&
 python prepare_dataset_oai_imo.py \
    ../../../data_raw/OAI_iMorphics_scans \
    ../../../data_raw/OAI_iMorphics_annotations \
    ../../../data/91_OAI_iMorphics_full_meta \
    --margin 20 \
)

(cd ../rocaseg/datasets &&
 echo "prepare_dataset" &&
 python prepare_dataset_okoa.py \
    ../../../data_raw/OKOA \
    ../../../data/31_OKOA_full_meta \
    --margin 0 \
)

(cd ../rocaseg/datasets &&
 echo "prepare_dataset" &&
 python prepare_dataset_maknee.py \
    ../../../data_raw/MAKNEE \
    ../../../data/41_MAKNEE_full_meta \
    --margin 0 \
)
# --------------------------------------------------------------------------------------

# -------------------------------- Resample datasets -----------------------------------
(cd ../rocaseg/ &&
 echo "resample" &&
 python resample.py \
    --path_root_in ../../data/31_OKOA_full_meta \
    --spacing_in 0.5859375 0.5859375 \
    --path_root_out ../../data/32_OKOA_full_meta_rescaled \
    --spacing_out 0.36458333 0.36458333 \
    --num_threads 12 \
    --margin 0 \
)

(cd ../rocaseg/ &&
 echo "resample" &&
 python resample.py \
    --path_root_in ../../data/41_MAKNEE_full_meta \
    --spacing_in 0.5859375 0.5859375 \
    --path_root_out ../../data/42_MAKNEE_full_meta_rescaled \
    --spacing_out 0.36458333 0.36458333 \
    --num_threads 12 \
    --margin 0 \
)
# --------------------------------------------------------------------------------------

# --------------------------------- Train models ---------------------------------------
(cd ../rocaseg/ &&
 echo "train" &&
 python train_baseline.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/0_baseline \
    --model_segm unet_lext \
    --input_channels 1 \
    --output_channels 5 \
    --center_depth 1 \
    --lr_segm 0.001 \
    --batch_size 32 \
    --epoch_num 50 \
    --fold_num 5 \
    --fold_idx -1 \
    --num_workers 12 \
 )

(cd ../rocaseg/ &&
 echo "train" &&
 python train_baseline.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/1_mixup \
    --model_segm unet_lext \
    --input_channels 1 \
    --output_channels 5 \
    --center_depth 1 \
    --lr_segm 0.001 \
    --batch_size 32 \
    --epoch_num 50 \
    --fold_num 5 \
    --fold_idx -1 \
    --with_mixup \
    --mixup_alpha 0.7 \
    --num_workers 12 \
 )

(cd ../rocaseg/ &&
 echo "train" &&
 python train_baseline.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/2_mixup_nowd \
    --model_segm unet_lext \
    --input_channels 1 \
    --output_channels 5 \
    --center_depth 1 \
    --lr_segm 0.001 \
    --wd_segm 0.0 \
    --batch_size 32 \
    --epoch_num 50 \
    --fold_num 5 \
    --fold_idx -1 \
    --with_mixup \
    --mixup_alpha 0.7 \
    --num_workers 12 \
 )

(cd ../rocaseg/ &&
 echo "train" &&
 python train_uda1.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/3_uda1 \
    --model_segm unet_lext \
    --center_depth 1 \
    --model_discr discriminator_a \
    --input_channels 1 \
    --output_channels 5 \
    --mask_mode all_unitibial_unimeniscus \
    --loss_segm multi_ce_loss \
    --lr_segm 0.0001 \
    --lr_discr 0.00004 \
    --batch_size 32 \
    --epoch_num 30 \
    --fold_num 5 \
    --fold_idx 0 \
    --num_workers 12 \
 )

(cd ../rocaseg/ &&
 echo "train" &&
 python train_uda1.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/4_uda2 \
    --model_segm unet_lext_aux \
    --center_depth 1 \
    --model_discr_out discriminator_a \
    --model_discr_aux discriminator_a \
    --input_channels 1 \
    --output_channels 5 \
    --mask_mode all_unitibial_unimeniscus \
    --loss_segm multi_ce_loss \
    --lr_segm 0.0001 \
    --lr_discr 0.00004 \
    --batch_size 32 \
    --epoch_num 30 \
    --fold_num 5 \
    --fold_idx 0 \
    --num_workers 12 \
 )

(cd ../rocaseg/ &&
 echo "train" &&
 python train_uda1.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/5_uda1_mixup_nowd \
    --model_segm unet_lext \
    --center_depth 1 \
    --model_discr discriminator_a \
    --input_channels 1 \
    --output_channels 5 \
    --mask_mode all_unitibial_unimeniscus \
    --loss_segm multi_ce_loss \
    --lr_segm 0.0001 \
    --lr_discr 0.00004 \
    --wd_segm 0.0 \
    --batch_size 32 \
    --epoch_num 30 \
    --fold_num 5 \
    --fold_idx 0 \
    --num_workers 12 \
    --with_mixup \
    --mixup_alpha 0.7 \
 )
# --------------------------------------------------------------------------------------

# ------------------------------- Run model inference ----------------------------------
for EXP in 0_baseline 1_mixup 2_mixup_nowd 3_uda1 5_uda1_mixup_nowd
do
(cd ../rocaseg/ &&
 echo "evaluate" &&
 python evaluate.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/${EXP} \
    --model_segm unet_lext \
    --center_depth 1 \
    --restore_weights \
    --output_channels 5 \
    --dataset oai_imo \
    --mask_mode all_unitibial_unimeniscus \
    --batch_size 64 \
    --fold_num 5 \
    --fold_idx -1 \
    --num_workers 12 \
    --predict_folds \
    --merge_predictions \
)
done

for EXP in 4_uda2
do
(cd ../rocaseg/ &&
 echo "evaluate" &&
 python evaluate.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/${EXP} \
    --model_segm unet_lext_aux \
    --center_depth 1 \
    --restore_weights \
    --output_channels 5 \
    --dataset oai_imo \
    --mask_mode all_unitibial_unimeniscus \
    --batch_size 64 \
    --fold_num 5 \
    --fold_idx -1 \
    --num_workers 12 \
    --predict_folds \
    --merge_predictions \
)
done

for EXP in 0_baseline 1_mixup 2_mixup_nowd 3_uda1 5_uda1_mixup_nowd
do
(cd ../rocaseg/ &&
 echo "evaluate" &&
 python evaluate.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/${EXP} \
    --model_segm unet_lext \
    --center_depth 1 \
    --restore_weights \
    --output_channels 5 \
    --dataset okoa \
    --mask_mode background_femoral_unitibial \
    --batch_size 64 \
    --fold_num 5 \
    --fold_idx -1 \
    --num_workers 12 \
    --predict_folds \
    --merge_predictions \
)
done

for EXP in 4_uda2
do
(cd ../rocaseg/ &&
 echo "evaluate" &&
 python evaluate.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/${EXP} \
    --model_segm unet_lext_aux \
    --center_depth 1 \
    --restore_weights \
    --output_channels 5 \
    --dataset okoa \
    --mask_mode background_femoral_unitibial \
    --batch_size 64 \
    --fold_num 5 \
    --fold_idx -1 \
    --num_workers 12 \
    --predict_folds \
    --merge_predictions \
)
done
# --------------------------------------------------------------------------------------

# ------------------------------- Analyze model predictions ----------------------------
for EXP in 0_baseline 1_mixup 2_mixup_nowd 3_uda1 4_uda2 5_uda1_mixup_nowd
do
(cd ../rocaseg/ &&
 echo "analyze_predictions" &&
 python analyze_predictions_single.py \
    --path_experiment_root ../../results/${EXP} \
    --dirname_pred mask_foldavg \
    --dirname_true mask_prep \
    --dataset oai_imo \
    --atlas segm \
    --ignore_cache \
    --num_workers 12 \
)
done

for EXP in 0_baseline 1_mixup 2_mixup_nowd 3_uda1 4_uda2 5_uda1_mixup_nowd
do
(cd ../rocaseg/ &&
 echo "analyze_predictions" &&
 python analyze_predictions_single.py \
    --path_experiment_root ../../results/${EXP} \
    --dirname_pred mask_foldavg \
    --dirname_true mask_prep \
    --dataset okoa \
    --atlas okoa \
    --ignore_cache \
    --num_workers 12 \
)
done
# --------------------------------------------------------------------------------------

# --------------------------------- Compare models -------------------------------------
(cd ../rocaseg/ &&
 echo "analyze_predictions_multi" &&
 python analyze_predictions_multi.py \
    --path_results_root ../../results \
    --experiment_id 0_baseline \
    --experiment_id 2_mixup_nowd \
    --experiment_id 4_uda2 \
    --dataset oai_imo \
    --atlas segm \
    --num_workers 12 \
)

(cd ../rocaseg/ &&
 echo "analyze_predictions_multi" &&
 python analyze_predictions_multi.py \
    --path_results_root ../../results \
    --experiment_id 0_baseline \
    --experiment_id 2_mixup_nowd \
    --experiment_id 4_uda2 \
    --dataset okoa \
    --atlas okoa \
    --num_workers 12 \
)
# --------------------------------------------------------------------------------------

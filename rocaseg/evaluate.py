import os
import logging
from glob import glob

import numpy as np
from skimage.color import label2rgb
from skimage import img_as_ubyte
from tqdm import tqdm
import click

import cv2
import tifffile
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from rocaseg.datasets import sources_from_path
from rocaseg.components import CheckpointHandler
from rocaseg.components.formats import numpy_to_nifti, png_to_numpy
from rocaseg.models import dict_models
from rocaseg.preproc import *
from rocaseg.repro import set_ultimate_seed


# The fix is a workaround to PyTorch multiprocessing issue:
# "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy('file_system')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logging.basicConfig()
logger = logging.getLogger('eval')
logger.setLevel(logging.INFO)

set_ultimate_seed()

if torch.cuda.is_available():
    maybe_gpu = 'cuda'
else:
    maybe_gpu = 'cpu'


def predict_folds(config, loader, fold_idcs):
    """Evaluate the model versus each fold
    """
    for fold_idx in fold_idcs:
        paths_weights_fold = dict()
        paths_weights_fold['segm'] = \
            os.path.join(config['path_weights'], 'segm', f'fold_{fold_idx}')

        handlers_ckpt = dict()
        handlers_ckpt['segm'] = CheckpointHandler(paths_weights_fold['segm'])

        paths_ckpt_sel = dict()
        paths_ckpt_sel['segm'] = handlers_ckpt['segm'].get_last_ckpt()

        # Initialize and configure the model
        model = (dict_models[config['model_segm']]
                 (input_channels=config['input_channels'],
                  output_channels=config['output_channels'],
                  center_depth=config['center_depth'],
                  pretrained=config['pretrained'],
                  restore_weights=config['restore_weights'],
                  path_weights=paths_ckpt_sel['segm']))
        model = nn.DataParallel(model).to(maybe_gpu)
        model.eval()

        with tqdm(total=len(loader), desc=f'Eval, fold {fold_idx}') as prog_bar:
            for i, data_batch in enumerate(loader):
                xs, ys_true = data_batch['xs'], data_batch['ys']
                xs, ys_true = xs.to(maybe_gpu), ys_true.to(maybe_gpu)

                if config['model_segm'] == 'unet_lext':
                    ys_pred = model(xs)
                elif config['model_segm'] == 'unet_lext_aux':
                    ys_pred, _ = model(xs)
                else:
                    msg = f"Unknown model {config['model_segm']}"
                    raise ValueError(msg)

                ys_pred_softmax = nn.Softmax(dim=1)(ys_pred)
                ys_pred_softmax_np = ys_pred_softmax.detach().to('cpu').numpy()

                data_batch['pred_softmax'] = ys_pred_softmax_np

                # Rearrange the batch
                data_dicts = [{k: v[n] for k, v in data_batch.items()}
                              for n in range(len(data_batch['image']))]

                for k, data_dict in enumerate(data_dicts):
                    dir_base = os.path.join(
                        config['path_predicts'],
                        data_dict['patient'], data_dict['release'], data_dict['sequence'])
                    fname_base = os.path.splitext(
                        os.path.basename(data_dict['path_rel_image']))[0]

                    # Save the predictions
                    dir_predicts = os.path.join(dir_base, 'mask_folds')
                    if not os.path.exists(dir_predicts):
                        os.makedirs(dir_predicts)

                    fname_full = os.path.join(
                        dir_predicts,
                        f'{fname_base}_fold_{fold_idx}.tiff')

                    tmp = (data_dict['pred_softmax'] * 255).astype(np.uint8, casting='unsafe')
                    tifffile.imsave(fname_full, tmp, compress=9)

                prog_bar.update(1)


def merge_predictions(config, source, loader, dict_fns,
                      save_plots=False, remove_foldw=False, convert_to_nifti=True):
    """Merge the predictions over all folds
    """
    dir_source_root = source['path_root']
    df_meta = loader.dataset.df_meta

    with tqdm(total=len(df_meta), desc='Merge') as prog_bar:
        for i, row in df_meta.iterrows():
            dir_scan_predicts = os.path.join(
                config['path_predicts'],
                row['patient'], row['release'], row['sequence'])
            dir_image_prep = os.path.join(dir_scan_predicts, 'image_prep')
            dir_mask_prep = os.path.join(dir_scan_predicts, 'mask_prep')
            dir_mask_folds = os.path.join(dir_scan_predicts, 'mask_folds')
            dir_mask_foldavg = os.path.join(dir_scan_predicts, 'mask_foldavg')
            dir_vis_foldavg = os.path.join(dir_scan_predicts, 'vis_foldavg')

            for p in (dir_image_prep, dir_mask_prep, dir_mask_folds, dir_mask_foldavg,
                      dir_vis_foldavg):
                if not os.path.exists(p):
                    os.makedirs(p)

            # Find the corresponding prediction files
            fname_base = os.path.splitext(os.path.basename(row['path_rel_image']))[0]

            fnames_pred = glob(os.path.join(dir_mask_folds, f'{fname_base}_fold_*.*'))

            # Read the reference data
            image = cv2.imread(
                os.path.join(dir_source_root, row['path_rel_image']),
                cv2.IMREAD_GRAYSCALE)
            image = dict_fns['crop'](image[None, ])[0]
            image = np.squeeze(image)
            if 'path_rel_mask' in row.index:
                ys_true = loader.dataset.read_mask(
                    os.path.join(dir_source_root, row['path_rel_mask']))
                if ys_true is not None:
                    ys_true = dict_fns['crop'](ys_true)[0]
            else:
                ys_true = None

            # Read the fold-wise predictions
            yss_pred = [tifffile.imread(f) for f in fnames_pred]
            ys_pred = np.stack(yss_pred, axis=0).astype(np.float32) / 255
            ys_pred = torch.from_numpy(ys_pred).unsqueeze(dim=0)

            # Average the fold predictions
            ys_pred = torch.mean(ys_pred, dim=1, keepdim=False)
            ys_pred_softmax = ys_pred / torch.sum(ys_pred, dim=1, keepdim=True)
            ys_pred_softmax_np = ys_pred_softmax.squeeze().numpy()

            ys_pred_arg_np = ys_pred_softmax_np.argmax(axis=0)

            # Save preprocessed input data
            fname_full = os.path.join(dir_image_prep, f'{fname_base}.png')
            cv2.imwrite(fname_full, image)  # image

            if ys_true is not None:
                ys_true = ys_true.astype(np.float32)
                ys_true = torch.from_numpy(ys_true).unsqueeze(dim=0)
                ys_true_arg_np = ys_true.numpy().squeeze().argmax(axis=0)
                fname_full = os.path.join(dir_mask_prep, f'{fname_base}.png')
                cv2.imwrite(fname_full, ys_true_arg_np)  # mask

            fname_meta = os.path.join(config['path_predicts'], 'meta_dynamic.csv')
            if not os.path.exists(fname_meta):
                df_meta.to_csv(fname_meta, index=False)  # metainfo

            # Save ensemble prediction
            fname_full = os.path.join(dir_mask_foldavg, f'{fname_base}.png')
            cv2.imwrite(fname_full, ys_pred_arg_np)

            # Save ensemble visualizations
            if save_plots:
                if ys_true is not None:
                    fname_full = os.path.join(
                        dir_vis_foldavg, f"{fname_base}_overlay_mask.png")
                    save_vis_overlay(image=image,
                                     mask=ys_true_arg_np,
                                     num_classes=config['output_channels'],
                                     fname=fname_full)

                fname_full = os.path.join(
                    dir_vis_foldavg, f"{fname_base}_overlay_pred.png")
                save_vis_overlay(image=image,
                                 mask=ys_pred_arg_np,
                                 num_classes=config['output_channels'],
                                 fname=fname_full)

                if ys_true is not None:
                    fname_full = os.path.join(
                        dir_vis_foldavg, f"{fname_base}_overlay_diff.png")
                    save_vis_mask_diff(image=image,
                                       mask_true=ys_true_arg_np,
                                       mask_pred=ys_pred_arg_np,
                                       fname=fname_full)

            # Remove the fold predictions
            if remove_foldw:
                for f in fnames_pred:
                    try:
                        os.remove(f)
                    except OSError:
                        logger.error(f'Cannot remove {f}')
            prog_bar.update(1)

    # Convert the results to 3D NIfTI images
    if convert_to_nifti:
        df_meta = df_meta.sort_values(by=["patient", "release", "sequence", "side"])

        for gb_name, gb_df in tqdm(
                df_meta.groupby(["patient", "release", "sequence", "side"]),
                desc="Convert to NIfTI"):

            patient, release, sequence, side = gb_name
            spacings = (gb_df['pixel_spacing_0'].iloc[0],
                        gb_df['pixel_spacing_1'].iloc[0],
                        gb_df['slice_thickness'].iloc[0])

            dir_scan_predicts = os.path.join(config['path_predicts'],
                                             patient, release, sequence)
            for result in ("image_prep", "mask_prep", "mask_foldavg"):
                pattern = os.path.join(dir_scan_predicts, result, '*.png')
                path_nii = os.path.join(dir_scan_predicts, f"{result}.nii")

                # Read and compose 3D image
                img = png_to_numpy(pattern_fname_in=pattern, reverse=False)

                # Save to NIfTI
                numpy_to_nifti(stack=img, fname_out=path_nii,
                               spacings=spacings, rcp_to_ras=True)


def save_vis_overlay(image, mask, num_classes, fname):
    # Add a sample of each class to have consistent class colors
    mask[0, :num_classes] = list(range(num_classes))
    overlay = label2rgb(label=mask, image=image, bg_label=0,
                        colors=['orangered', 'gold', 'lime', 'fuchsia'])
    # Convert to uint8 to save space
    overlay = img_as_ubyte(overlay)
    # Save to file
    if overlay.ndim == 3:
        overlay = overlay[:, :, ::-1]
    cv2.imwrite(fname, overlay)


def save_vis_mask_diff(image, mask_true, mask_pred, fname):
    diff = np.empty_like(mask_true)
    diff[(mask_true == mask_pred) & (mask_pred == 0)] = 0  # TN
    diff[(mask_true == mask_pred) & (mask_pred != 0)] = 0  # TP
    diff[(mask_true != mask_pred) & (mask_pred == 0)] = 2  # FP
    diff[(mask_true != mask_pred) & (mask_pred != 0)] = 3  # FN
    diff_colors = ('green', 'red', 'yellow')
    diff[0, :4] = [0, 1, 2, 3]
    overlay = label2rgb(label=diff, image=image, bg_label=0,
                        colors=diff_colors)
    # Convert to uint8 to save space
    overlay = img_as_ubyte(overlay)
    # Save to file
    if overlay.ndim == 3:
        overlay = overlay[:, :, ::-1]
    cv2.imwrite(fname, overlay)


@click.command()
@click.option('--path_data_root', default='../../data')
@click.option('--path_experiment_root', default='../../results/temporary')
@click.option('--model_segm', default='unet_lext')
@click.option('--center_depth', default=1, type=int)
@click.option('--pretrained', is_flag=True)
@click.option('--restore_weights', is_flag=True)
@click.option('--input_channels', default=1, type=int)
@click.option('--output_channels', default=1, type=int)
@click.option('--dataset', type=click.Choice(
    ['oai_imo', 'okoa', 'maknee']))
@click.option('--subset',  type=click.Choice(
    ['test', 'all']))
@click.option('--mask_mode', default='all_unitibial_unimeniscus', type=str)
@click.option('--sample_mode', default='x_y', type=str)
@click.option('--batch_size', default=64, type=int)
@click.option('--fold_num', default=5, type=int)
@click.option('--fold_idx', default=-1, type=int)
@click.option('--fold_idx_ignore', multiple=True, type=int)
@click.option('--num_workers', default=1, type=int)
@click.option('--seed_trainval_test', default=0, type=int)
@click.option('--predict_folds', is_flag=True)
@click.option('--merge_predictions', is_flag=True)
@click.option('--save_plots', is_flag=True)
def main(**config):
    config['path_data_root'] = os.path.abspath(config['path_data_root'])
    config['path_experiment_root'] = os.path.abspath(config['path_experiment_root'])

    config['path_weights'] = os.path.join(config['path_experiment_root'], 'weights')
    if not os.path.exists(config['path_weights']):
        raise ValueError('{} does not exist'.format(config['path_weights']))

    config['path_predicts'] = os.path.join(
        config['path_experiment_root'], f"predicts_{config['dataset']}_test")
    config['path_logs'] = os.path.join(
        config['path_experiment_root'], f"logs_{config['dataset']}_test")

    os.makedirs(config['path_predicts'], exist_ok=True)
    os.makedirs(config['path_logs'], exist_ok=True)

    logging_fh = logging.FileHandler(
        os.path.join(config['path_logs'], 'main.log'))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)

    # Collect the available and specified sources
    sources = sources_from_path(path_data_root=config['path_data_root'],
                                selection=config['dataset'],
                                with_folds=True,
                                seed_trainval_test=config['seed_trainval_test'])

    # Select the subset for evaluation
    if config['subset'] == 'test':
        logging.warning('Using the regular trainval-test split')
    elif config['subset'] == 'all':
        logging.warning('Using data selection: full dataset')
        for s in sources:
            sources[s]['test_df'] = sources[s]['sel_df']
            logger.info(f"Selected number of samples: {len(sources[s]['test_df'])}")
    else:
        raise ValueError(f"Unknown dataset: {config['subset']}")

    if config['dataset'] == 'oai_imo':
        from rocaseg.datasets import DatasetOAIiMoSagittal2d as DatasetSagittal2d
    elif config['dataset'] == 'okoa':
        from rocaseg.datasets import DatasetOKOASagittal2d as DatasetSagittal2d
    elif config['dataset'] == 'maknee':
        from rocaseg.datasets import DatasetMAKNEESagittal2d as DatasetSagittal2d
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    # Configure dataset-dependent transforms
    fn_crop = CenterCrop(height=300, width=300)
    if config['dataset'] == 'oai_imo':
        fn_norm = Normalize(mean=0.252699, std=0.251142)
        fn_unnorm = UnNormalize(mean=0.252699, std=0.251142)
    elif config['dataset'] == 'okoa':
        fn_norm = Normalize(mean=0.232454, std=0.236259)
        fn_unnorm = UnNormalize(mean=0.232454, std=0.236259)
    else:
        msg = f"No transforms defined for dataset: {config['dataset']}"
        raise NotImplementedError(msg)
    dict_fns = {'crop': fn_crop, 'norm': fn_norm, 'unnorm': fn_unnorm}

    dataset_test = DatasetSagittal2d(
        df_meta=sources[config['dataset']]['test_df'], mask_mode=config['mask_mode'],
        name=config['dataset'], sample_mode=config['sample_mode'],
        transforms=[
            PercentileClippingAndToFloat(cut_min=10, cut_max=99),
            fn_crop,
            fn_norm,
            ToTensor()
        ])
    loader_test = DataLoader(dataset_test,
                             batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers'],
                             drop_last=False)

    # Build a list of folds to run on
    if config['fold_idx'] == -1:
        fold_idcs = list(range(config['fold_num']))
    else:
        fold_idcs = [config['fold_idx'], ]
    for g in config['fold_idx_ignore']:
        fold_idcs = [i for i in fold_idcs if i != g]

    # Execute
    with torch.no_grad():
        if config['predict_folds']:
            predict_folds(config=config, loader=loader_test, fold_idcs=fold_idcs)

        if config['merge_predictions']:
            merge_predictions(config=config, source=sources[config['dataset']],
                              loader=loader_test, dict_fns=dict_fns,
                              save_plots=config['save_plots'], remove_foldw=False,
                              convert_to_nifti=True)


if __name__ == '__main__':
    main()

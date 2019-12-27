import os
import pickle
from tqdm import tqdm
import click
import logging
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from rocaseg.components import dice_score
from rocaseg.components.formats import png_to_numpy
from rocaseg.datasets.constants import atlas_to_locations


logging.basicConfig()
logger = logging.getLogger('analyze')
logger.setLevel(logging.INFO)


def metrics_paired_slicew(path_pred, dirname_true, dirname_pred,
                          df, num_classes, num_workers=1):
    def _process_3d_pair(path_root, name_true, name_pred, meta):
        # Read the data
        if meta['side'] == 'RIGHT':
            reverse = True
        else:
            reverse = False

        patt_true = os.path.join(path_root, meta['patient'], meta['release'],
                                 meta['sequence'], name_true, '*.png')
        stack_true = png_to_numpy(patt_true, reverse=reverse)

        patt_pred = os.path.join(path_root, meta['patient'], meta['release'],
                                 meta['sequence'], name_pred, '*.png')
        stack_pred = png_to_numpy(patt_pred, reverse=reverse)
        if stack_true.shape != stack_pred.shape:
            msg = (f'Reference and predictions samples are of different shape: '
                   f'{name_true}: {stack_true.shape}, {name_pred}: {stack_pred.shape}')
            raise ValueError(msg)
        num_slices = stack_true.shape[-1]

        # Add batch dimension
        stack_true = stack_true[None, ...]
        stack_pred = stack_pred[None, ...]

        # Compute the metrics
        res = []
        for slice_idx in range(num_slices):
            tmp = {
                'dice_score': dice_score(stack_pred[..., slice_idx],
                                         stack_true[..., slice_idx],
                                         num_classes=num_classes),
                'slice_idx_proc': slice_idx,
                **meta,
            }
            res.append(tmp)
        return res

    acc_ls = []
    groupers_stack = ['patient', 'release', 'sequence', 'side']

    for name_gb, df_gb in tqdm(df.groupby(groupers_stack)):
        patient, release, sequence, side = name_gb

        acc_ls.append(delayed(_process_3d_pair)(
            path_root=path_pred,
            name_true=dirname_true,
            name_pred=dirname_pred,
            meta={
                'patient': patient,
                'release': release,
                'sequence': sequence,
                'side': side,
                **df_gb.to_dict("records")[0],
            }
        ))

    acc_ls = Parallel(n_jobs=num_workers, verbose=1)(acc_ls)
    acc_l = []
    for acc in acc_ls:
        acc_l.extend(acc)

    # Convert from list of dicts to dict of lists
    acc_d = {k: [d[k] for d in acc_l] for k in acc_l[0]}
    return acc_d


def plot_metrics_paired_slicew_xvd7(acc, path_root_out, num_classes, class_names,
                                    metric_names, config):
    """Average and visualize the metrics.

    Args:
        acc: dict of lists
        path_root_out: str
        num_classes: int
        class_names: dict
        metric_names: iterable of str
        config: dict

    """
    for metric_name in metric_names:
        if metric_name not in acc:
            logger.error(f'`{metric_name}` is not presented in `acc`')
            continue
        metric_values = acc[metric_name]

        ncols = 2
        nrows = (num_classes - 1) // ncols + 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
        axes = axes.ravel()

        tmp = np.stack(metric_values, axis=-1)  # shape: (class_idx, sample_idx)
        tmp_means, tmp_stds = [], []

        for class_idx, class_scores in enumerate(tmp[1:], start=1):
            class_idx_axes = class_idx - 1
            class_scores = class_scores[class_scores != 0]

            if np.any(np.isnan(class_scores)):
                logger.warning('NaN score')

            tmp_means.append(np.mean(class_scores))
            tmp_stds.append(np.std(class_scores))

            axes[class_idx_axes].hist(class_scores, bins=40,
                                      range=(0.3, 1.0), density=True)
            axes[class_idx_axes].set_title(class_names[class_idx])
            axes[class_idx_axes].set_ylim((0, 10))

        tmp_values = ', '.join(
            [f'{m:.03f}({s:.03f})' for m, s in zip(tmp_means, tmp_stds)])
        logger.info(f"{metric_name}:\n{tmp_values}\n")

        plt.tight_layout()
        fname_vis = f"metrics_{config['dataset']}_test_slicew_{metric_name}.png"
        path_vis = os.path.join(path_root_out, fname_vis)
        plt.savefig(path_vis)
        if config['interactive']:
            plt.show()
        else:
            plt.close()


def print_metrics_paired_slicew_pua5(acc, metric_names):
    """Average and print the metrics with respect to KL.

    Args:
        acc: dict of lists
        metric_names: iterable of str

    """
    kl_vec = np.asarray(acc['KL'])

    for kl_value in np.unique(kl_vec):
        kl_sel = kl_vec == kl_value

        for metric_name in metric_names:
            if metric_name not in acc:
                logger.error(f'`{metric_name}` is not presented in `acc`')
                continue
            metric_values = acc[metric_name]
            tmp = np.stack(metric_values, axis=-1)[..., kl_sel]
            # shape: (class_idx, sample_idx)

            tmp_means, tmp_stds = [], []
            for class_idx, class_scores in enumerate(tmp[1:], start=1):
                class_scores = class_scores[class_scores != 0]

                tmp_means.append(np.mean(class_scores))
                tmp_stds.append(np.std(class_scores))

            tmp_values = ', '.join(
                [f'{m:.03f}({s:.03f})' for m, s in zip(tmp_means, tmp_stds)])
            logger.info(f"KL{kl_value}, {metric_name}:\n{tmp_values}\n")


def plot_metrics_paired_slicew_va49(acc, path_root_out, num_classes, class_names,
                                    metric_names, config):
    """Average and visualize the metrics.

    Args:
        acc:
        path_root_out:
        num_classes:
        class_names:
        metric_names:
        config:

    """
    groupers_stack = ['patient', 'release', 'sequence', 'side']

    acc_df = pd.DataFrame.from_dict(acc)
    acc_df = acc_df.sort_values([*groupers_stack, 'slice_idx_proc'])

    for metric_name in metric_names:
        if metric_name not in acc:
            logger.error(f'`{metric_name}` is not presented in `acc`')
            continue

        metric_values = []
        for gb_name, gb_df in acc_df.groupby(groupers_stack):
            tmp = np.asarray([np.asarray(e) for e in gb_df[metric_name]])
            metric_values.append(tmp)
        metric_values = np.stack(metric_values, axis=0)
        # Axes order: (scan, slice_idx, class_idx)

        nrows = np.floor(np.sqrt(num_classes)).astype(int)
        ncols = np.ceil(float(num_classes) / nrows).astype(int)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
        ax = axes.ravel()
        plt.title(metric_name)

        for class_idx in range(1, num_classes):
            class_idx_axes = class_idx - 1

            y = metric_values[..., class_idx]
            x = np.tile(np.arange(0, y.shape[1]), reps=(y.shape[0], 1))
            y = y.ravel()
            x = x.ravel()

            sel = (y != 0)
            y = y[sel]
            x = x[sel]

            sns.lineplot(x=x, y=y, err_style='band', ax=ax[class_idx_axes],
                         color="salmon")
            if class_idx != 0:
                ax[class_idx_axes].set_ylim((0.4, 1))
            ax[class_idx_axes].set_xlim((10, 150))
            ax[class_idx_axes].set_xlabel('Slice index')  #, size=16)
            ax[class_idx_axes].set_ylabel('DSC')  #, size=16)
            ax[class_idx_axes].set_title(class_names[class_idx])  #, size=16)

            # for label in (ax[class_idx_axes].get_xticklabels() +
            #               ax[class_idx_axes].get_yticklabels()):
            #     label.set_fontsize(15)

        plt.tight_layout()

        fname_vis = f"metrics_{config['dataset']}_test_slicew_confid_{metric_name}.png"
        path_vis = os.path.join(path_root_out, fname_vis)
        plt.savefig(path_vis)
        if config['interactive']:
            plt.show()
        else:
            plt.close()


def _scan_to_metrics_paired(path_root, name_true, name_pred, num_classes, meta):
    # Read the data
    patt_true = os.path.join(path_root, meta['patient'], meta['release'],
                             meta['sequence'], name_true, '*.png')
    stack_true = png_to_numpy(patt_true)

    patt_pred = os.path.join(path_root, meta['patient'], meta['release'],
                             meta['sequence'], name_pred, '*.png')
    stack_pred = png_to_numpy(patt_pred)

    # Add batch dimension
    stack_true = stack_true[None, ...]
    stack_pred = stack_pred[None, ...]

    # Compute the metrics
    res = meta
    res['dice_score'] = dice_score(stack_pred, stack_true,
                                   num_classes=num_classes)
    return res


def metrics_paired_volumew(*, path_pred, dirname_true, dirname_pred, df, num_classes,
                           num_workers=1):
    acc_l = []
    groupers_stack = ['patient', 'release', 'sequence', 'side']

    for name_gb, df_gb in tqdm(df.groupby(groupers_stack)):
        patient, release, sequence, side = name_gb

        acc_l.append(delayed(_scan_to_metrics_paired)(
            path_root=path_pred,
            name_true=dirname_true,
            name_pred=dirname_pred,
            num_classes=num_classes,
            meta={
                'patient': patient,
                'release': release,
                'sequence': sequence,
                'side': side,
                **df_gb.to_dict("records")[0],
            }
        ))

    acc_l = Parallel(n_jobs=num_workers, verbose=10)(
        t for t in tqdm(acc_l, total=len(acc_l)))
    # Convert from list of dicts to dict of lists
    acc_d = {k: [d[k] for d in acc_l] for k in acc_l[0]}
    return acc_d


def plot_metrics_paired_volumew_n5a9(acc, path_root_out, num_classes, class_names,
                                     metric_names, config):
    """Average and visualize the metrics.

    Args:
        acc: dict of lists
        path_root_out: str
        num_classes: int
        class_names: dict
        metric_names: iterable of str
        config: dict

    """
    groupers_stack = ['patient', 'release', 'sequence', 'side']

    acc_df = pd.DataFrame.from_dict(acc)
    acc_df = acc_df.sort_values(groupers_stack)

    for metric_name in metric_names:
        if metric_name not in acc:
            logger.error(f'`{metric_name}` is not presented in `acc`')
            continue

        metric_values = []
        for gb_name, gb_df in acc_df.groupby(groupers_stack):
            tmp = np.asarray([np.asarray(e) for e in gb_df[metric_name]])
            metric_values.append(tmp)
        metric_values = np.stack(metric_values, axis=0)
        # Axes order: (scan, class_idx[, sub_metric])

        nrows = np.floor(np.sqrt(num_classes)).astype(int)
        ncols = np.ceil(float(num_classes) / nrows).astype(int)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
        ax = axes.ravel()

        tmp_means, tmp_stds = [], []

        xlims = {
            'dice_score': (0.5, 1.0),
        }

        for class_idx, class_scores in enumerate(metric_values.T[1:], start=1):
            class_idx_axes = class_idx - 1

            if metric_name == 'dice_score':
                class_scores = class_scores[class_scores != 0]
            class_scores = np.squeeze(class_scores)

            tmp_means.append(np.mean(class_scores))
            tmp_stds.append(np.std(class_scores))

            if metric_name in xlims:
                ax[class_idx_axes].hist(class_scores, bins=50,
                                        range=xlims[metric_name],
                                        density=True)
            else:
                ax[class_idx_axes].hist(class_scores, bins=50,
                                        density=True)
            ax[class_idx_axes].set_title(class_names[class_idx])

        tmp_values = ', '.join(
            [f'{m:.03f}({s:.03f})' for m, s in zip(tmp_means, tmp_stds)])
        logger.info(f"{metric_name}:\n{tmp_values}\n")

        plt.tight_layout()
        fname_vis = f"metrics_{config['dataset']}_test_volumew_{metric_name}.png"
        path_vis = os.path.join(path_root_out, fname_vis)
        plt.savefig(path_vis)
        if config['interactive']:
            plt.show()
        else:
            plt.close()


def print_metrics_paired_volumew_b46e(acc, metric_names):
    """Average and print the metrics with respect to KL grades.

    Args:
        acc: dict of lists
        metric_names: tuple of str

    """
    kl_vec = np.asarray(acc['KL'])

    for kl_value in np.unique(kl_vec):
        kl_sel = kl_vec == kl_value

        for metric_name in metric_names:
            if metric_name not in acc:
                logger.error(f'`{metric_name}` is not presented in `acc`')
                continue
            metric_values = acc[metric_name]
            tmp = np.asarray(metric_values)[kl_sel]

            tmp_means, tmp_stds = [], []
            for class_idx, class_scores in enumerate(
                    np.moveaxis(tmp[..., 1:], -1, 0), start=1):
                class_scores = class_scores[class_scores != 0]

                tmp_means.append(np.mean(class_scores))
                tmp_stds.append(np.std(class_scores))

            tmp_values = ', '.join(
                [f'{m:.03f}({s:.03f})' for m, s in zip(tmp_means, tmp_stds)])
            logger.info(f"KL{kl_value}, {metric_name}\n{tmp_values}\n")


@click.command()
@click.option('--path_experiment_root', default='../../results/temporary')
@click.option('--dirname_pred', required=True)
@click.option('--dirname_true', required=True)
@click.option('--dataset', required=True, type=click.Choice(
    ['oai_imo', 'okoa', 'maknee']))
@click.option('--atlas', required=True, type=click.Choice(
    ['imo', 'segm', 'okoa']))
@click.option('--ignore_cache', is_flag=True)
@click.option('--interactive', is_flag=True)
@click.option('--num_workers', default=1, type=int)
def main(**config):

    path_pred = os.path.join(config['path_experiment_root'],
                             f"predicts_{config['dataset']}_test")
    path_logs = os.path.join(config['path_experiment_root'],
                             f"logs_{config['dataset']}_test")

    # Get the information on object classes
    locations = atlas_to_locations[config['atlas']]
    class_names = [k for k in locations]
    num_classes = max(locations.values()) + 1

    # Get the index of image files and the corresponding metadata
    path_meta = os.path.join(path_pred, 'meta_dynamic.csv')
    df_meta = pd.read_csv(path_meta,
                          dtype={'patient': str,
                                 'release': str,
                                 'prefix_var': str,
                                 'sequence': str,
                                 'side': str,
                                 'slice_idx': int,
                                 'pixel_spacing_0': float,
                                 'pixel_spacing_1': float,
                                 'slice_thickness': float,
                                 'KL': int,
                                 'has_mask': int},
                          index_col=False)

    df_sel = df_meta.sort_values(['patient', 'release', 'sequence', 'side', 'slice_idx'])

    # -------------------------------- Planar ------------------------------------------
    fname_pkl = os.path.join(path_logs,
                             f"cache_{config['dataset']}_test_"
                             f"{config['atlas']}_"
                             f"slicew_paired.pkl")

    logger.info('Planar scores')
    if os.path.exists(fname_pkl) and not config['ignore_cache']:
        logger.info('Loading from the cache')
        with open(fname_pkl, 'rb') as f:
            acc_slicew = pickle.load(f)
    else:
        logger.info('Computing')
        acc_slicew = metrics_paired_slicew(path_pred=path_pred,
                                           dirname_true=config['dirname_true'],
                                           dirname_pred=config['dirname_pred'],
                                           df=df_sel,
                                           num_classes=num_classes,
                                           num_workers=config['num_workers'])
        logger.info('Caching the results into file')
        os.makedirs(path_logs, exist_ok=True)
        with open(fname_pkl, 'wb') as f:
            pickle.dump(acc_slicew, f)

    plot_metrics_paired_slicew_xvd7(
        acc=acc_slicew,
        path_root_out=path_logs,
        num_classes=num_classes,
        class_names=class_names,
        metric_names=('dice_score', ),
        config=config,
    )

    print_metrics_paired_slicew_pua5(
        acc=acc_slicew,
        metric_names=('dice_score',),
    )

    plot_metrics_paired_slicew_va49(
        acc=acc_slicew,
        path_root_out=path_logs,
        num_classes=num_classes,
        class_names=class_names,
        metric_names=('dice_score',),
        config=config,
    )

    # ------------------------------- Volumetric ---------------------------------------
    fname_pkl = os.path.join(path_logs,
                             f"cache_{config['dataset']}_test_"
                             f"{config['atlas']}_"
                             f"volumew_paired.pkl")

    logger.info('Volumetric scores')
    if os.path.exists(fname_pkl) and not config['ignore_cache']:
        logger.info('Loading from the cache')
        with open(fname_pkl, 'rb') as f:
            acc_volumew = pickle.load(f)
    else:
        logger.info('Computing')
        acc_volumew = metrics_paired_volumew(
            path_pred=path_pred,
            dirname_true=config['dirname_true'],
            dirname_pred=config['dirname_pred'],
            df=df_sel,
            num_classes=num_classes,
            num_workers=config['num_workers']
        )
        logger.info('Caching the results into file')
        os.makedirs(path_logs, exist_ok=True)
        with open(fname_pkl, 'wb') as f:
            pickle.dump(acc_volumew, f)

    plot_metrics_paired_volumew_n5a9(
        acc=acc_volumew,
        path_root_out=path_logs,
        num_classes=num_classes,
        class_names=class_names,
        metric_names=('dice_score', ),
        config=config,
    )

    print_metrics_paired_volumew_b46e(
        acc=acc_volumew,
        metric_names=('dice_score', ),
    )


if __name__ == '__main__':
    main()

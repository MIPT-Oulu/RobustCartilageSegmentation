import os
import pickle
from collections import OrderedDict

import click
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from rocaseg.datasets.constants import atlas_to_locations

logging.basicConfig()
logger = logging.getLogger('analyze')
logger.setLevel(logging.INFO)


def slicew_dsc_distr_vs_slice_idcs(results, config, num_classes, class_names):
    """Distribution of planar DSCs VS slice indices
    """
    path_vis = config['path_results_root']

    metric_names = ['dice_score', ]
    # colors = ["darkorange", "mediumorchid", "deepskyblue"]
    # colors = ["salmon", "dodgerblue", "orangered"]
    labels = ['Baseline', '+ mixup - WD', '+ UDA2']
    colors = ["lightsalmon", "dodgerblue", 'grey']
    # labels = ['Reference', '+ mixup - WD']

    # Average and visualize the metrics
    for metric_name in metric_names:
        print(metric_name)

        for class_idx in range(1, num_classes):

            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 3.6))
            plt.title(metric_name)

            for acc_idx, (_, acc_slicew) in enumerate(results.items()):
                unique_scans = list(set(zip(
                    acc_slicew['patient'],
                    acc_slicew['release']
                )))
                num_scans = len(unique_scans)
                num_slices = 160

                y = np.full((num_slices, num_scans), np.nan)
                for slice_idx in range(num_slices):
                    sel_by_slice = np.asarray(acc_slicew['slice_idx_proc']) == slice_idx
                    scores = np.asarray(acc_slicew['dice_score'])[sel_by_slice]
                    scores = scores[:, class_idx]

                    y[slice_idx, :] = scores

                x = np.ones_like(y) * np.arange(0, num_slices)[:, None]
                y = y.ravel()
                x = x.ravel()

                sel = (y != 0)
                y = y[sel]
                x = x[sel]

                sns.lineplot(x=x, y=y, err_style='band', ax=axes,
                             color=colors[acc_idx], label=labels[acc_idx])

                fontsize = 14
                axes.set_ylim((0.45, 0.95))
                # axes.set_ylim((0.5, 1.0))
                axes.set_xlim((10, 150))
                axes.set_xlabel('slice index', size=fontsize)
                axes.set_ylabel('DSC', size=fontsize)
                tmp_title = class_names[class_idx]
                tmp_title = tmp_title.replace('femoral', 'Femoral cartilage')
                tmp_title = tmp_title.replace('tibial', 'Tibial cartilage')
                tmp_title = tmp_title.replace('patellar', 'Patellar cartilage')
                tmp_title = tmp_title.replace('menisci', 'Menisci')

                axes.set_title(tmp_title, size=fontsize)

                for label in (axes.get_xticklabels() + axes.get_yticklabels()):
                    label.set_fontsize(fontsize)

                leg = axes.legend(
                    # loc='lower left',
                    loc='lower right',
                    prop={'size': fontsize}, ncol=1,
                    framealpha=1.0,
                )
                for line in leg.get_lines():
                    line.set_linewidth(3.0)

                # axes[class_idx_axes].get_legend().set_visible(False)

            plt.grid(linestyle=':')

            fname_vis = os.path.join(path_vis,
                                     f"metrics_{config['dataset']}_"
                                     f"test_slicew_confid_"
                                     f"{metric_name}_{class_idx}.pdf")
            plt.savefig(fname_vis, bbox_inches='tight')
            logger.info(f"Saved to {fname_vis}")
            if config['interactive']:
                plt.show()
            else:
                plt.close()


@click.command()
@click.option('--path_results_root', default='../../results')
@click.option('--experiment_id', multiple=True)
@click.option('--dataset', required=True, type=click.Choice(
    ['oai_imo', 'okoa', 'maknee']))
@click.option('--atlas', required=True, type=click.Choice(
    ['imo', 'segm', 'okoa']))
@click.option('--interactive', is_flag=True)
@click.option('--num_workers', default=1, type=int)
def main(**config):
    results_slicew = OrderedDict()
    results_volumew = OrderedDict()

    for exp_id in config['experiment_id']:
        path_experiment_root = os.path.join(
            config['path_results_root'], exp_id)
        path_logs = os.path.join(
            path_experiment_root, f"logs_{config['dataset']}_test")

        # Get the information on object classes
        locations = atlas_to_locations[config['atlas']]
        class_names = [k for k in locations]
        num_classes = max(locations.values()) + 1

        # Load precomputed planar scores
        fname_slicew = os.path.join(
            path_logs,
            f"cache_{config['dataset']}_test_{config['atlas']}_slicew_paired.pkl"
        )
        if os.path.exists(fname_slicew):
            with open(fname_slicew, 'rb') as f:
                acc_slicew = pickle.load(f)
        else:
            raise IOError(f'File {fname_slicew} does not exist')

        fname_volumew = os.path.join(
            path_logs,
            f"cache_{config['dataset']}_test_{config['atlas']}_volumew_paired.pkl"
        )
        if os.path.exists(fname_volumew):
            with open(fname_volumew, 'rb') as f:
                acc_volumew = pickle.load(f)
        else:
            raise IOError(f'File {fname_volumew} does not exist')

        results_slicew.update({exp_id: acc_slicew})
        results_volumew.update({exp_id: acc_volumew})

    # ------------------------------ Visualization --------------------------------------

    slicew_dsc_distr_vs_slice_idcs(results=results_slicew,
                                   config=config,
                                   num_classes=num_classes,
                                   class_names=class_names)


if __name__ == '__main__':
    main()

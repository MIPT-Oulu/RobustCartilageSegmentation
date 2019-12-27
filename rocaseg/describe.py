import os
import logging
from collections import defaultdict

import click

import torch
from torch.utils.data.dataloader import DataLoader

from rocaseg.datasets import sources_from_path
from rocaseg.preproc import *
from rocaseg.repro import set_ultimate_seed


logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

set_ultimate_seed()

if torch.cuda.is_available():
    maybe_gpu = 'cuda'
else:
    maybe_gpu = 'cpu'


class Describer:
    def __init__(self, config):
        self.config = config

    def run(self, loader):
        metrics_avg = defaultdict(float)

        for i, data_batch in enumerate(loader):
            metrics_curr = dict()
            xs, ys_true = data_batch
            # xs, ys_true = xs.to(maybe_gpu), ys_true.to(maybe_gpu)

            # Calculate metrics
            with torch.no_grad():
                e = self.config['metrics_skip_edge']
                if e != 0:
                    metrics_curr['mean'] = xs[:, :, e:-e, e:-e].mean()
                    metrics_curr['std'] = xs[:, :, e:-e, e:-e].std()
                    metrics_curr['var'] = xs[:, :, e:-e, e:-e].var()
                else:
                    metrics_curr['mean'] = xs.mean()
                    metrics_curr['std'] = xs.std()
                    metrics_curr['var'] = xs.var()
            for k, v in metrics_curr.items():
                metrics_avg[k] += v

        # Add metrics logging
        logger.info('Metrics:')
        metrics_avg = {k: v / len(loader)
                       for k, v in metrics_avg.items()}
        for k, v in metrics_avg.items():
            logger.info(f'{k}: {v}')


@click.command()
@click.option('--path_data_root', default='../../data')
@click.option('--path_experiment_root', default='../../results/temporary')
@click.option('--dataset', type=click.Choice(
    ['oai_imo', 'okoa', 'maknee']))
@click.option('--mask_mode', default='all_unitibial_unimeniscus', type=str)
@click.option('--sample_mode', default='x_y', type=str)
@click.option('--batch_size', default=64, type=int)
@click.option('--num_workers', default=1, type=int)
@click.option('--seed_trainval_test', default=0, type=int)
@click.option('--metrics_skip_edge', default=0, type=int)
def main(**config):
    config['path_logs'] = os.path.join(
        config['path_experiment_root'], f"logs_{config['dataset']}_describe")

    os.makedirs(config['path_logs'], exist_ok=True)

    logging_fh = logging.FileHandler(
        os.path.join(config['path_logs'], 'main.log'))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)

    # Collect the available and specified sources
    sources = sources_from_path(path_data_root=config['path_data_root'],
                                selection=config['dataset'],
                                with_folds=False,
                                seed_trainval_test=config['seed_trainval_test'])

    if config['dataset'] == 'oai_imo':
        from rocaseg.datasets import DatasetOAIiMoSagittal2d as DatasetSagittal2d
    elif config['dataset'] == 'okoa':
        from rocaseg.datasets import DatasetOKOASagittal2d as DatasetSagittal2d
    elif config['dataset'] == 'maknee':
        from rocaseg.datasets import DatasetMAKNEESagittal2d as DatasetSagittal2d
    else:
        raise ValueError('Unknown dataset')

    for subset in ('trainval', 'test'):
        name = subset
        df = sources[config['dataset']][f"{subset}_df"]

        dataset = DatasetSagittal2d(
            df_meta=df, mask_mode=config['mask_mode'], name=name,
            sample_mode=config['sample_mode'],
            transforms=[
                PercentileClippingAndToFloat(cut_min=10, cut_max=99),
                ToTensor()
        ])

        loader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            drop_last=False)
        describer = Describer(config=config)

        describer.run(loader)
        loader.dataset.describe()


if __name__ == '__main__':
    main()

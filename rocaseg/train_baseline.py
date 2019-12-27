import os
import logging
from collections import defaultdict
import click

import numpy as np
import cv2

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rocaseg.datasets import DatasetOAIiMoSagittal2d, sources_from_path
from rocaseg.models import dict_models
from rocaseg.components import (dict_losses, confusion_matrix, dice_score_from_cm,
                                dict_optimizers, CheckpointHandler)
from rocaseg.preproc import *
from rocaseg.repro import set_ultimate_seed
from rocaseg.components.mixup import mixup_criterion, mixup_data


cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

logging.basicConfig()
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)

set_ultimate_seed()

if torch.cuda.is_available():
    maybe_gpu = 'cuda'
else:
    maybe_gpu = 'cpu'


class ModelTrainer:
    def __init__(self, config, fold_idx=None):
        self.config = config
        self.fold_idx = fold_idx

        self.paths_weights_fold = dict()
        self.paths_weights_fold['segm'] = \
            os.path.join(config['path_weights'], 'segm', f'fold_{self.fold_idx}')
        os.makedirs(self.paths_weights_fold['segm'], exist_ok=True)

        self.path_logs_fold = \
            os.path.join(config['path_logs'], f'fold_{self.fold_idx}')
        os.makedirs(self.path_logs_fold, exist_ok=True)

        self.handlers_ckpt = dict()
        self.handlers_ckpt['segm'] = CheckpointHandler(self.paths_weights_fold['segm'])

        paths_ckpt_sel = dict()
        paths_ckpt_sel['segm'] = self.handlers_ckpt['segm'].get_last_ckpt()

        # Initialize and configure the models
        self.models = dict()
        self.models['segm'] = (dict_models[config['model_segm']]
                               (input_channels=self.config['input_channels'],
                                output_channels=self.config['output_channels'],
                                center_depth=self.config['center_depth'],
                                pretrained=self.config['pretrained'],
                                path_pretrained=self.config['path_pretrained_segm'],
                                restore_weights=self.config['restore_weights'],
                                path_weights=paths_ckpt_sel['segm']))
        self.models['segm'] = nn.DataParallel(self.models['segm'])
        self.models['segm'] = self.models['segm'].to(maybe_gpu)

        # Configure the training
        self.optimizers = dict()
        self.optimizers['segm'] = (dict_optimizers['adam'](
            self.models['segm'].parameters(),
            lr=self.config['lr_segm'],
            weight_decay=self.config['wd_segm']))

        self.lr_update_rule = {30: 0.1}

        self.losses = dict()
        self.losses['segm'] = dict_losses[self.config['loss_segm']](
            num_classes=self.config['output_channels'],
        )

        self.losses['segm'] = self.losses['segm'].to(maybe_gpu)

        self.tensorboard = SummaryWriter(self.path_logs_fold)

    def run_one_epoch(self, epoch_idx, loaders):
        name_ds = list(loaders.keys())[0]

        fnames_acc = defaultdict(list)
        metrics_acc = dict()
        metrics_acc['samplew'] = defaultdict(list)
        metrics_acc['batchw'] = defaultdict(list)
        metrics_acc['datasetw'] = defaultdict(list)
        metrics_acc['datasetw'][f'{name_ds}__cm'] = \
            np.zeros((self.config['output_channels'],) * 2, dtype=np.uint32)

        prog_bar_params = {'postfix': {'epoch': epoch_idx}, }

        if self.models['segm'].training:
            # ------------------------ Training regime ------------------------
            loader_ds = loaders[name_ds]['train']

            steps_ds = len(loader_ds)
            prog_bar_params.update({'total': steps_ds,
                                    'desc': f'Train, epoch {epoch_idx}'})

            loader_ds_iter = iter(loader_ds)

            with tqdm(**prog_bar_params) as prog_bar:
                for step_idx in range(steps_ds):
                    self.optimizers['segm'].zero_grad()

                    data_batch_ds = next(loader_ds_iter)

                    xs_ds, ys_true_ds = data_batch_ds['xs'], data_batch_ds['ys']
                    fnames_acc['oai'].extend(data_batch_ds['path_image'])

                    ys_true_arg_ds = torch.argmax(ys_true_ds.long(), dim=1)
                    xs_ds = xs_ds.to(maybe_gpu)
                    ys_true_arg_ds = ys_true_arg_ds.to(maybe_gpu)

                    if not self.config['with_mixup']:
                        ys_pred_ds = self.models['segm'](xs_ds)

                        loss_segm = self.losses['segm'](input_=ys_pred_ds,
                                                        target=ys_true_arg_ds)
                    else:
                        xs_mixup, ys_mixup_a, ys_mixup_b, lambda_mixup = mixup_data(
                            x=xs_ds, y=ys_true_arg_ds,
                            alpha=self.config['mixup_alpha'], device=maybe_gpu)
                        ys_pred_ds = self.models['segm'](xs_mixup)
                        loss_segm = mixup_criterion(criterion=self.losses['segm'],
                                                    pred=ys_pred_ds,
                                                    y_a=ys_mixup_a,
                                                    y_b=ys_mixup_b,
                                                    lam=lambda_mixup)

                    metrics_acc['batchw']['loss'].append(loss_segm.item())

                    loss_segm.backward()
                    self.optimizers['segm'].step()

                    prog_bar.update(1)
        else:
            # ----------------------- Validation regime -----------------------
            loader_ds = loaders[name_ds]['val']

            steps_ds = len(loader_ds)
            prog_bar_params.update({'total': steps_ds,
                                    'desc': f'Validate, epoch {epoch_idx}'})

            loader_ds_iter = iter(loader_ds)

            with torch.no_grad(), tqdm(**prog_bar_params) as prog_bar:
                for step_idx in range(steps_ds):
                    data_batch_ds = next(loader_ds_iter)

                    xs_ds, ys_true_ds = data_batch_ds['xs'], data_batch_ds['ys']
                    fnames_acc['oai'].extend(data_batch_ds['path_image'])

                    ys_true_arg_ds = torch.argmax(ys_true_ds.long(), dim=1)
                    xs_ds = xs_ds.to(maybe_gpu)
                    ys_true_arg_ds = ys_true_arg_ds.to(maybe_gpu)

                    if not self.config['with_mixup']:
                        ys_pred_ds = self.models['segm'](xs_ds)
                        loss_segm = self.losses['segm'](input_=ys_pred_ds,
                                                        target=ys_true_arg_ds)
                    else:
                        xs_mixup, ys_mixup_a, ys_mixup_b, lambda_mixup = mixup_data(
                            x=xs_ds, y=ys_true_arg_ds,
                            alpha=self.config['mixup_alpha'], device=maybe_gpu)

                        ys_pred_ds = self.models['segm'](xs_mixup)
                        loss_segm = mixup_criterion(criterion=self.losses['segm'],
                                                    pred=ys_pred_ds,
                                                    y_a=ys_mixup_a,
                                                    y_b=ys_mixup_b,
                                                    lam=lambda_mixup)

                    metrics_acc['batchw']['loss'].append(loss_segm.item())

                    # Calculate metrics
                    ys_pred_softmax_ds = nn.Softmax(dim=1)(ys_pred_ds)
                    ys_pred_softmax_np_ds = ys_pred_softmax_ds.to('cpu').numpy()

                    ys_pred_arg_np_ds = ys_pred_softmax_np_ds.argmax(axis=1)
                    ys_true_arg_np_ds = ys_true_arg_ds.to('cpu').numpy()

                    metrics_acc['datasetw'][f'{name_ds}__cm'] += confusion_matrix(
                        ys_pred_arg_np_ds, ys_true_arg_np_ds,
                        self.config['output_channels'])

                    prog_bar.update(1)

        for k, v in metrics_acc['samplew'].items():
            metrics_acc['samplew'][k] = np.asarray(v)
        metrics_acc['datasetw'][f'{name_ds}__dice_score'] = np.asarray(
            dice_score_from_cm(metrics_acc['datasetw'][f'{name_ds}__cm']))
        return metrics_acc, fnames_acc

    def fit(self, loaders):
        epoch_idx_best = -1
        loss_best = float('inf')
        metrics_train_best = dict()
        fnames_train_best = []
        metrics_val_best = dict()
        fnames_val_best = []

        for epoch_idx in range(self.config['epoch_num']):
            self.models = {n: m.train() for n, m in self.models.items()}
            metrics_train, fnames_train = \
                self.run_one_epoch(epoch_idx, loaders)

            # Process the accumulated metrics
            for k, v in metrics_train['batchw'].items():
                if k.startswith('loss'):
                    metrics_train['datasetw'][k] = np.mean(np.asarray(v))
                else:
                    logger.warning(f'Non-processed batch-wise entry: {k}')

            self.models = {n: m.eval() for n, m in self.models.items()}
            metrics_val, fnames_val = \
                self.run_one_epoch(epoch_idx, loaders)

            # Process the accumulated metrics
            for k, v in metrics_val['batchw'].items():
                if k.startswith('loss'):
                    metrics_val['datasetw'][k] = np.mean(np.asarray(v))
                else:
                    logger.warning(f'Non-processed batch-wise entry: {k}')

            # Learning rate update
            for s, m in self.lr_update_rule.items():
                if epoch_idx == s:
                    for name, optim in self.optimizers.items():
                        for param_group in optim.param_groups:
                            param_group['lr'] *= m

            # Add console logging
            logger.info(f'Epoch: {epoch_idx}')
            for subset, metrics in (('train', metrics_train),
                                    ('val', metrics_val)):
                logger.info(f'{subset} metrics:')
                for k, v in metrics['datasetw'].items():
                    logger.info(f'{k}: \n{v}')

            # Add TensorBoard logging
            for subset, metrics in (('train', metrics_train),
                                    ('val', metrics_val)):
                # Log only dataset-reduced metrics
                for k, v in metrics['datasetw'].items():
                    if isinstance(v, np.ndarray):
                        self.tensorboard.add_scalars(
                            f'fold_{self.fold_idx}/{k}_{subset}',
                            {f'class{i}': e for i, e in enumerate(v.ravel().tolist())},
                            global_step=epoch_idx)
                    elif isinstance(v, (str, int, float)):
                        self.tensorboard.add_scalar(
                            f'fold_{self.fold_idx}/{k}_{subset}',
                            float(v),
                            global_step=epoch_idx)
                    else:
                        logger.warning(f'{k} is of unsupported dtype {v}')
            for name, optim in self.optimizers.items():
                for param_group in optim.param_groups:
                    self.tensorboard.add_scalar(
                        f'fold_{self.fold_idx}/learning_rate/{name}',
                        param_group['lr'],
                        global_step=epoch_idx)

            # Save the model
            loss_curr = metrics_val['datasetw']['loss']
            if loss_curr < loss_best:
                loss_best = loss_curr
                epoch_idx_best = epoch_idx
                metrics_train_best = metrics_train
                metrics_val_best = metrics_val
                fnames_train_best = fnames_train
                fnames_val_best = fnames_val

                self.handlers_ckpt['segm'].save_new_ckpt(
                    model=self.models['segm'],
                    model_name=self.config['model_segm'],
                    fold_idx=self.fold_idx,
                    epoch_idx=epoch_idx)

        msg = (f'Finished fold {self.fold_idx} '
               f'with the best loss {loss_best:.5f} '
               f'on epoch {epoch_idx_best}, '
               f'weights: ({self.paths_weights_fold})')
        logger.info(msg)
        return (metrics_train_best, fnames_train_best,
                metrics_val_best, fnames_val_best)


@click.command()
@click.option('--path_data_root', default='../../data')
@click.option('--path_experiment_root', default='../../results/temporary')
@click.option('--model_segm', default='unet_lext')
@click.option('--center_depth', default=1, type=int)
@click.option('--pretrained', is_flag=True)
@click.option('--path_pretrained_segm', type=str, help='Path to .pth file')
@click.option('--restore_weights', is_flag=True)
@click.option('--input_channels', default=1, type=int)
@click.option('--output_channels', default=1, type=int)
@click.option('--dataset', type=click.Choice(
    ['oai_imo', 'okoa', 'maknee']), default='oai_imo')
@click.option('--mask_mode', default='all_unitibial_unimeniscus', type=str)
@click.option('--sample_mode', default='x_y', type=str)
@click.option('--loss_segm', default='multi_ce_loss')
@click.option('--lr_segm', default=0.0001, type=float)
@click.option('--wd_segm', default=5e-5, type=float)
@click.option('--optimizer_segm', default='adam')
@click.option('--batch_size', default=64, type=int)
@click.option('--epoch_size', default=1.0, type=float)
@click.option('--epoch_num', default=2, type=int)
@click.option('--fold_num', default=5, type=int)
@click.option('--fold_idx', default=-1, type=int)
@click.option('--fold_idx_ignore', multiple=True, type=int)
@click.option('--num_workers', default=1, type=int)
@click.option('--seed_trainval_test', default=0, type=int)
@click.option('--with_mixup', is_flag=True)
@click.option('--mixup_alpha', default=1, type=float)
def main(**config):
    config['path_data_root'] = os.path.abspath(config['path_data_root'])
    config['path_experiment_root'] = os.path.abspath(config['path_experiment_root'])

    config['path_weights'] = os.path.join(config['path_experiment_root'], 'weights')
    config['path_logs'] = os.path.join(config['path_experiment_root'], 'logs_train')
    os.makedirs(config['path_weights'], exist_ok=True)
    os.makedirs(config['path_logs'], exist_ok=True)

    logging_fh = logging.FileHandler(
        os.path.join(config['path_logs'], 'main_{}.log'.format(config['fold_idx'])))
    logging_fh.setLevel(logging.DEBUG)
    logger.addHandler(logging_fh)

    # Collect the available and specified sources
    sources = sources_from_path(path_data_root=config['path_data_root'],
                                selection=config['dataset'],
                                with_folds=True,
                                fold_num=config['fold_num'],
                                seed_trainval_test=config['seed_trainval_test'])

    # Build a list of folds to run on
    if config['fold_idx'] == -1:
        fold_idcs = list(range(config['fold_num']))
    else:
        fold_idcs = [config['fold_idx'], ]
    for g in config['fold_idx_ignore']:
        fold_idcs = [i for i in fold_idcs if i != g]

    # Train each fold separately
    fold_scores = dict()

    # Use straightforward fold allocation strategy
    folds = list(sources[config['dataset']]['trainval_folds'])

    for fold_idx, idcs_subsets in enumerate(folds):
        if fold_idx not in fold_idcs:
            continue
        logger.info(f'Training fold {fold_idx}')

        name_ds = config['dataset']

        (sources[name_ds]['train_idcs'], sources[name_ds]['val_idcs']) = idcs_subsets

        sources[name_ds]['train_df'] = \
            sources[name_ds]['trainval_df'].iloc[sources[name_ds]['train_idcs']]
        sources[name_ds]['val_df'] = \
            sources[name_ds]['trainval_df'].iloc[sources[name_ds]['val_idcs']]

        for n, s in sources.items():
            logger.info('Made {} train-val split, number of samples: {}, {}'
                        .format(n, len(s['train_df']), len(s['val_df'])))

        datasets = defaultdict(dict)

        datasets[name_ds]['train'] = DatasetOAIiMoSagittal2d(
            df_meta=sources[name_ds]['train_df'],
            mask_mode=config['mask_mode'],
            sample_mode=config['sample_mode'],
            transforms=[
                PercentileClippingAndToFloat(cut_min=10, cut_max=99),
                CenterCrop(height=300, width=300),
                HorizontalFlip(prob=.5),
                GammaCorrection(gamma_range=(0.5, 1.5), prob=.5),
                OneOf([
                    DualCompose([
                        Scale(ratio_range=(0.7, 0.8), prob=1.),
                        Scale(ratio_range=(1.5, 1.6), prob=1.),
                    ]),
                    NoTransform()
                ]),
                Crop(output_size=(300, 300)),
                BilateralFilter(d=5, sigma_color=50, sigma_space=50, prob=.3),
                Normalize(mean=0.252699, std=0.251142),
                ToTensor(),
        ])
        datasets[name_ds]['val'] = DatasetOAIiMoSagittal2d(
            df_meta=sources[name_ds]['val_df'],
            mask_mode=config['mask_mode'],
            sample_mode=config['sample_mode'],
            transforms=[
                PercentileClippingAndToFloat(cut_min=10, cut_max=99),
                CenterCrop(height=300, width=300),
                Normalize(mean=0.252699, std=0.251142),
                ToTensor()
        ])

        loaders = defaultdict(dict)

        loaders[name_ds]['train'] = DataLoader(
            datasets[name_ds]['train'],
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True)
        loaders[name_ds]['val'] = DataLoader(
            datasets[name_ds]['val'],
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=True)

        trainer = ModelTrainer(config=config, fold_idx=fold_idx)

        # INFO: run once before the training to compute the dataset statistics
        # dataset_train.describe()

        tmp = trainer.fit(loaders=loaders)
        metrics_train, fnames_train, metrics_val, fnames_val = tmp

        fold_scores[fold_idx] = (metrics_val['datasetw'][f'{name_ds}__dice_score'], )

        trainer.tensorboard.close()
    logger.info(f'Fold scores:\n{repr(fold_scores)}')


if __name__ == '__main__':
    main()

import os
import logging
from collections import defaultdict
import gc
import click
import resource

import numpy as np
import cv2

import torch
import torch.nn.functional as torch_fn
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rocaseg.datasets import (DatasetOAIiMoSagittal2d,
                              DatasetOKOASagittal2d,
                              DatasetMAKNEESagittal2d,
                              sources_from_path)
from rocaseg.models import dict_models
from rocaseg.components import (dict_losses, confusion_matrix, dice_score_from_cm,
                                dict_optimizers, CheckpointHandler)
from rocaseg.preproc import *
from rocaseg.repro import set_ultimate_seed


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

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
        self.paths_weights_fold['discr_out'] = \
            os.path.join(config['path_weights'], 'discr_out', f'fold_{self.fold_idx}')
        os.makedirs(self.paths_weights_fold['discr_out'], exist_ok=True)
        self.paths_weights_fold['discr_aux'] = \
            os.path.join(config['path_weights'], 'discr_aux', f'fold_{self.fold_idx}')
        os.makedirs(self.paths_weights_fold['discr_aux'], exist_ok=True)

        self.path_logs_fold = \
            os.path.join(config['path_logs'], f'fold_{self.fold_idx}')
        os.makedirs(self.path_logs_fold, exist_ok=True)

        self.handlers_ckpt = dict()
        self.handlers_ckpt['segm'] = CheckpointHandler(self.paths_weights_fold['segm'])
        self.handlers_ckpt['discr_aux'] = CheckpointHandler(self.paths_weights_fold['discr_aux'])
        self.handlers_ckpt['discr_out'] = CheckpointHandler(self.paths_weights_fold['discr_out'])

        paths_ckpt_sel = dict()
        paths_ckpt_sel['segm'] = self.handlers_ckpt['segm'].get_last_ckpt()
        paths_ckpt_sel['discr_aux'] = self.handlers_ckpt['discr_aux'].get_last_ckpt()
        paths_ckpt_sel['discr_out'] = self.handlers_ckpt['discr_out'].get_last_ckpt()

        # Initialize and configure the models
        self.models = dict()
        self.models['segm'] = (dict_models[config['model_segm']]
                               (input_channels=self.config['input_channels'],
                                output_channels=self.config['output_channels'],
                                center_depth=self.config['center_depth'],
                                pretrained=self.config['pretrained'],
                                path_pretrained=self.config['path_pretrained_segm'],
                                restore_weights=self.config['restore_weights'],
                                path_weights=paths_ckpt_sel['segm'],
                                with_aux=True))
        self.models['segm'] = nn.DataParallel(self.models['segm'])
        self.models['segm'] = self.models['segm'].to(maybe_gpu)

        self.models['discr_aux'] = (dict_models[config['model_discr_aux']]
                                    (input_channels=self.config['output_channels'],
                                     output_channels=1,
                                     pretrained=self.config['pretrained'],
                                     restore_weights=self.config['restore_weights'],
                                     path_weights=paths_ckpt_sel['discr_aux']))
        self.models['discr_aux'] = nn.DataParallel(self.models['discr_aux'])
        self.models['discr_aux'] = self.models['discr_aux'].to(maybe_gpu)

        self.models['discr_out'] = (dict_models[config['model_discr_out']]
                                    (input_channels=self.config['output_channels'],
                                     output_channels=1,
                                     pretrained=self.config['pretrained'],
                                     restore_weights=self.config['restore_weights'],
                                     path_weights=paths_ckpt_sel['discr_out']))
        self.models['discr_out'] = nn.DataParallel(self.models['discr_out'])
        self.models['discr_out'] = self.models['discr_out'].to(maybe_gpu)

        # Configure the training
        self.optimizers = dict()
        self.optimizers['segm'] = (dict_optimizers['adam'](
            self.models['segm'].parameters(),
            lr=self.config['lr_segm'],
            weight_decay=5e-5))
        self.optimizers['discr_aux'] = (dict_optimizers['adam'](
            self.models['discr_aux'].parameters(),
            lr=self.config['lr_discr'],
            weight_decay=5e-5))
        self.optimizers['discr_out'] = (dict_optimizers['adam'](
            self.models['discr_out'].parameters(),
            lr=self.config['lr_discr'],
            weight_decay=5e-5))

        self.lr_update_rule = {25: 0.1}

        self.losses = dict()
        self.losses['segm'] = dict_losses[self.config['loss_segm']](
            num_classes=self.config['output_channels'],
        )
        self.losses['advers'] = dict_losses['bce_loss']()
        self.losses['discr'] = dict_losses['bce_loss']()

        self.losses['segm'] = self.losses['segm'].to(maybe_gpu)
        self.losses['advers'] = self.losses['advers'].to(maybe_gpu)
        self.losses['discr'] = self.losses['discr'].to(maybe_gpu)

        self.tensorboard = SummaryWriter(self.path_logs_fold)

    def run_one_epoch(self, epoch_idx, loaders):
        COEFF_DISCR = 1
        COEFF_SEGM_AUX = 0.1
        COEFF_SEGM_OUT = 1
        COEFF_ADVERS_AUX = 0.0002
        COEFF_ADVERS_OUT = 0.001

        fnames_acc = defaultdict(list)
        metrics_acc = dict()
        metrics_acc['samplew'] = defaultdict(list)
        metrics_acc['batchw'] = defaultdict(list)
        metrics_acc['datasetw'] = defaultdict(list)
        metrics_acc['datasetw']['cm_oai'] = \
            np.zeros((self.config['output_channels'],) * 2, dtype=np.uint32)
        metrics_acc['datasetw']['cm_okoa'] = \
            np.zeros((self.config['output_channels'],) * 2, dtype=np.uint32)

        prog_bar_params = {'postfix': {'epoch': epoch_idx}, }

        is_training = (self.models['segm'].training and
                       self.models['discr_aux'].training and
                       self.models['discr_out'].training)
        if is_training:
            # ------------------------ Training regime ------------------------
            loader_oai = loaders['oai_imo']['train']
            loader_maknee = loaders['maknee']['train']

            steps_oai, steps_maknee = len(loader_oai), len(loader_maknee)
            steps_total = steps_oai
            prog_bar_params.update({'total': steps_total,
                                    'desc': f'Train, epoch {epoch_idx}'})

            loader_oai_iter = iter(loader_oai)
            loader_maknee_iter = iter(loader_maknee)

            loader_oai_iter_old = None
            loader_maknee_iter_old = None

            with tqdm(**prog_bar_params) as prog_bar:
                for step_idx in range(steps_total):
                    self.optimizers['segm'].zero_grad()
                    self.optimizers['discr_aux'].zero_grad()
                    self.optimizers['discr_out'].zero_grad()

                    metrics_acc['batchw']['loss_total'].append(0)

                    try:
                        data_batch_oai = next(loader_oai_iter)
                    except StopIteration:
                        loader_oai_iter_old = loader_oai_iter
                        loader_oai_iter = iter(loader_oai)
                        data_batch_oai = next(loader_oai_iter)

                    try:
                        data_batch_maknee = next(loader_maknee_iter)
                    except StopIteration:
                        loader_maknee_iter_old = loader_maknee_iter
                        loader_maknee_iter = iter(loader_maknee)
                        data_batch_maknee = next(loader_maknee_iter)

                    xs_oai, ys_true_oai = data_batch_oai['xs'], data_batch_oai['ys']
                    fnames_acc['oai'].extend(data_batch_oai['path_image'])
                    xs_oai = xs_oai.to(maybe_gpu)
                    ys_true_arg_oai = torch.argmax(ys_true_oai.long().to(maybe_gpu), dim=1)

                    xs_maknee, ys_true_maknee = data_batch_maknee['xs'], data_batch_maknee['ys']
                    fnames_acc['maknee'].extend(data_batch_maknee['path_image'])
                    xs_maknee = xs_maknee.to(maybe_gpu)

                    # -------------- Train discriminator network -------------
                    # With source
                    ys_pred_out_oai, ys_pred_aux_oai = self.models['segm'](xs_oai)
                    ys_pred_out_softmax_oai = torch_fn.softmax(ys_pred_out_oai, dim=1)
                    ys_pred_aux_softmax_oai = torch_fn.softmax(ys_pred_aux_oai, dim=1)

                    zs_pred_out_oai = self.models['discr_out'](ys_pred_out_softmax_oai)
                    zs_pred_aux_oai = self.models['discr_aux'](ys_pred_aux_softmax_oai)

                    # Use 0 as a label for the source domain
                    # Output representations
                    loss_discr_out_0 = self.losses['discr'](
                        input=zs_pred_out_oai,
                        target=torch.zeros_like(zs_pred_out_oai, device=maybe_gpu))
                    loss_discr_out_0 = loss_discr_out_0 / 2 * COEFF_DISCR
                    loss_discr_out_0.backward(retain_graph=True)
                    metrics_acc['batchw']['loss_discr_out_0'].append(loss_discr_out_0.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_discr_out_0'][-1]

                    # Penultimate representations
                    loss_discr_aux_0 = self.losses['discr'](
                        input=zs_pred_aux_oai,
                        target=torch.zeros_like(zs_pred_aux_oai, device=maybe_gpu))
                    loss_discr_aux_0 = loss_discr_aux_0 / 2 * COEFF_DISCR
                    loss_discr_aux_0.backward(retain_graph=True)
                    metrics_acc['batchw']['loss_discr_aux_0'].append(
                        loss_discr_aux_0.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_discr_aux_0'][-1]

                    # With target
                    self.models['segm'] = self.models['segm'].eval()
                    ys_pred_out_maknee, ys_pred_aux_maknee = self.models['segm'](xs_maknee)
                    self.models['segm'] = self.models['segm'].train()

                    ys_pred_out_softmax_maknee = torch_fn.softmax(ys_pred_out_maknee, dim=1)
                    ys_pred_aux_softmax_maknee = torch_fn.softmax(ys_pred_aux_maknee, dim=1)
                    zs_pred_out_maknee = self.models['discr_out'](ys_pred_out_softmax_maknee)
                    zs_pred_aux_maknee = self.models['discr_aux'](ys_pred_aux_softmax_maknee)

                    # Use 1 as a label for the target domain
                    # Output representations
                    loss_discr_out_1 = self.losses['discr'](
                        input=zs_pred_out_maknee,
                        target=torch.ones_like(zs_pred_out_maknee, device=maybe_gpu))
                    loss_discr_out_1 = loss_discr_out_1 / 2 * COEFF_DISCR
                    loss_discr_out_1.backward(retain_graph=True)
                    metrics_acc['batchw']['loss_discr_out_1'].append(loss_discr_out_1.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_discr_out_1'][-1]

                    # Penultimate representations
                    loss_discr_aux_1 = self.losses['discr'](
                        input=zs_pred_aux_maknee,
                        target=torch.ones_like(zs_pred_aux_maknee, device=maybe_gpu))
                    loss_discr_aux_1 = loss_discr_aux_1 / 2 * COEFF_DISCR
                    loss_discr_aux_1.backward()
                    metrics_acc['batchw']['loss_discr_aux_1'].append(
                        loss_discr_aux_1.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_discr_aux_1'][-1]

                    self.models['segm'].zero_grad()
                    self.optimizers['discr_aux'].step()
                    self.optimizers['discr_out'].step()
                    self.models['discr_aux'].zero_grad()
                    self.models['discr_out'].zero_grad()

                    # ---------------- Train segmentation network ------------
                    # With source
                    ys_pred_out_oai, ys_pred_aux_oai = self.models['segm'](xs_oai)

                    # Output representations
                    loss_segm_out = self.losses['segm'](input_=ys_pred_out_oai,
                                                        target=ys_true_arg_oai)
                    loss_segm_out.backward(retain_graph=True)
                    loss_segm_out = loss_segm_out * COEFF_SEGM_OUT
                    metrics_acc['batchw']['loss_segm_out'].append(loss_segm_out.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_segm_out'][-1]

                    # Penultimate representations
                    loss_segm_aux = self.losses['segm'](input_=ys_pred_aux_oai,
                                                        target=ys_true_arg_oai)
                    loss_segm_aux.backward(retain_graph=True)
                    loss_segm_aux = loss_segm_aux * COEFF_SEGM_AUX
                    metrics_acc['batchw']['loss_segm_aux'].append(loss_segm_aux.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_segm_aux'][-1]

                    # With target
                    self.models['segm'] = self.models['segm'].eval()
                    ys_pred_out_maknee, ys_pred_aux_maknee = self.models['segm'](xs_maknee)
                    self.models['segm'] = self.models['segm'].train()

                    ys_pred_out_softmax_maknee = torch_fn.softmax(ys_pred_out_maknee, dim=1)
                    ys_pred_aux_softmax_maknee = torch_fn.softmax(ys_pred_aux_maknee, dim=1)
                    zs_pred_out_maknee = self.models['discr_out'](ys_pred_out_softmax_maknee)
                    zs_pred_aux_maknee = self.models['discr_aux'](ys_pred_aux_softmax_maknee)

                    # Use 0 as a label for the source domain
                    # Output representations
                    loss_advers_out = self.losses['advers'](
                        input=zs_pred_out_maknee,
                        target=torch.zeros_like(zs_pred_out_maknee, device=maybe_gpu))
                    loss_advers_out = loss_advers_out * COEFF_ADVERS_OUT
                    loss_advers_out.backward(retain_graph=True)
                    metrics_acc['batchw']['loss_advers_out'].append(loss_advers_out.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_advers_out'][-1]

                    # Penultimate representations
                    loss_advers_aux = self.losses['advers'](
                        input=zs_pred_aux_maknee,
                        target=torch.zeros_like(zs_pred_aux_maknee, device=maybe_gpu))
                    loss_advers_aux = loss_advers_aux * COEFF_ADVERS_AUX
                    loss_advers_aux.backward()
                    metrics_acc['batchw']['loss_advers_aux'].append(
                        loss_advers_aux.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_advers_aux'][-1]

                    self.models['discr_out'].zero_grad()
                    self.models['discr_aux'].zero_grad()
                    self.optimizers['segm'].step()
                    self.models['segm'].zero_grad()

                    if step_idx % 10 == 0:
                        self.tensorboard.add_scalars(
                            f'fold_{self.fold_idx}/losses_train',
                            {'discr_0_aux_batchw':
                                 metrics_acc['batchw']['loss_discr_aux_0'][-1],
                             'discr_0_out_batchw':
                                 metrics_acc['batchw']['loss_discr_out_0'][-1],
                             'discr_1_aux_batchw':
                                 metrics_acc['batchw']['loss_discr_aux_1'][-1],
                             'discr_1_out_batchw':
                                 metrics_acc['batchw']['loss_discr_out_1'][-1],
                             'discr_sum_aux_batchw':
                                 (metrics_acc['batchw']['loss_discr_aux_0'][-1] +
                                  metrics_acc['batchw']['loss_discr_aux_1'][-1]),
                             'discr_sum_out_batchw':
                                 (metrics_acc['batchw']['loss_discr_out_0'][-1] +
                                  metrics_acc['batchw']['loss_discr_out_1'][-1]),
                             'discr_sum_all_batchw':
                                 (metrics_acc['batchw']['loss_discr_aux_0'][-1] +
                                  metrics_acc['batchw']['loss_discr_out_0'][-1] +
                                  metrics_acc['batchw']['loss_discr_aux_1'][-1] +
                                  metrics_acc['batchw']['loss_discr_out_1'][-1]),
                             'segm_aux_batchw': metrics_acc['batchw']['loss_segm_aux'][-1],
                             'segm_out_batchw': metrics_acc['batchw']['loss_segm_out'][-1],
                             'advers_aux_batchw': metrics_acc['batchw']['loss_advers_aux'][-1],
                             'advers_out_batchw': metrics_acc['batchw']['loss_advers_out'][-1],
                             'total_batchw': metrics_acc['batchw']['loss_total'][-1],
                             }, global_step=(epoch_idx * steps_total + step_idx))

                    prog_bar.update(1)

            del [loader_oai_iter_old, loader_maknee_iter_old]
            gc.collect()
        else:
            # ----------------------- Validation regime -----------------------
            loader_oai = loaders['oai_imo']['val']
            loader_okoa = loaders['okoa']['val']
            loader_maknee = loaders['maknee']['val']

            steps_oai, steps_okoa, steps_maknee = len(loader_oai), len(loader_okoa), len(loader_maknee)
            steps_total = steps_oai
            prog_bar_params.update({'total': steps_total,
                                    'desc': f'Validate, epoch {epoch_idx}'})

            loader_oai_iter = iter(loader_oai)
            loader_okoa_iter = iter(loader_okoa)
            loader_maknee_iter = iter(loader_maknee)

            loader_oai_iter_old = None
            loader_okoa_iter_old = None
            loader_maknee_iter_old = None

            with torch.no_grad(), tqdm(**prog_bar_params) as prog_bar:
                for step_idx in range(steps_total):
                    metrics_acc['batchw']['loss_total'].append(0)

                    try:
                        data_batch_oai = next(loader_oai_iter)
                    except StopIteration:
                        loader_oai_iter_old = loader_oai_iter
                        loader_oai_iter = iter(loader_oai)
                        data_batch_oai = next(loader_oai_iter)

                    try:
                        data_batch_okoa = next(loader_okoa_iter)
                    except StopIteration:
                        loader_okoa_iter_old = loader_okoa_iter
                        loader_okoa_iter = iter(loader_okoa)
                        data_batch_okoa = next(loader_okoa_iter)

                    try:
                        data_batch_maknee = next(loader_maknee_iter)
                    except StopIteration:
                        loader_maknee_iter_old = loader_maknee_iter
                        loader_maknee_iter = iter(loader_maknee)
                        data_batch_maknee = next(loader_maknee_iter)

                    xs_oai, ys_true_oai = data_batch_oai['xs'], data_batch_oai['ys']
                    fnames_acc['oai'].extend(data_batch_oai['path_image'])
                    xs_oai = xs_oai.to(maybe_gpu)
                    ys_true_arg_oai = torch.argmax(ys_true_oai.long().to(maybe_gpu), dim=1)

                    xs_maknee, ys_true_maknee = data_batch_maknee['xs'], data_batch_maknee['ys']
                    fnames_acc['maknee'].extend(data_batch_maknee['path_image'])
                    xs_maknee = xs_maknee.to(maybe_gpu)

                    # -------------- Validate discriminator network -------------
                    # With source
                    ys_pred_out_oai, ys_pred_aux_oai = self.models['segm'](xs_oai)
                    ys_pred_out_softmax_oai = torch_fn.softmax(ys_pred_out_oai, dim=1)
                    ys_pred_aux_softmax_oai = torch_fn.softmax(ys_pred_aux_oai, dim=1)

                    zs_pred_out_oai = self.models['discr_out'](ys_pred_out_softmax_oai)
                    zs_pred_aux_oai = self.models['discr_aux'](ys_pred_aux_softmax_oai)

                    # Use 0 as a label for the source domain
                    # Output representations
                    loss_discr_out_0 = self.losses['discr'](
                        input=zs_pred_out_oai,
                        target=torch.zeros_like(zs_pred_out_oai, device=maybe_gpu))
                    loss_discr_out_0 = loss_discr_out_0 / 2 * COEFF_DISCR
                    metrics_acc['batchw']['loss_discr_out_0'].append(
                        loss_discr_out_0.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_discr_out_0'][-1]

                    # Penultimate representations
                    loss_discr_aux_0 = self.losses['discr'](
                        input=zs_pred_aux_oai,
                        target=torch.zeros_like(zs_pred_aux_oai, device=maybe_gpu))
                    loss_discr_aux_0 = loss_discr_aux_0 / 2 * COEFF_DISCR
                    metrics_acc['batchw']['loss_discr_aux_0'].append(
                        loss_discr_aux_0.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_discr_aux_0'][-1]

                    # With target
                    ys_pred_out_maknee, ys_pred_aux_maknee = self.models['segm'](xs_maknee)

                    ys_pred_out_softmax_maknee = torch_fn.softmax(ys_pred_out_maknee,
                                                                  dim=1)
                    ys_pred_aux_softmax_maknee = torch_fn.softmax(ys_pred_aux_maknee,
                                                                  dim=1)
                    zs_pred_out_maknee = self.models['discr_out'](ys_pred_out_softmax_maknee)
                    zs_pred_aux_maknee = self.models['discr_aux'](ys_pred_aux_softmax_maknee)

                    # Use 1 as a label for the target domain
                    # Output representations
                    loss_discr_out_1 = self.losses['discr'](
                        input=zs_pred_out_maknee,
                        target=torch.ones_like(zs_pred_out_maknee, device=maybe_gpu))
                    loss_discr_out_1 = loss_discr_out_1 / 2 * COEFF_DISCR
                    metrics_acc['batchw']['loss_discr_out_1'].append(
                        loss_discr_out_1.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_discr_out_1'][-1]

                    # Penultimate representations
                    loss_discr_aux_1 = self.losses['discr'](
                        input=zs_pred_aux_maknee,
                        target=torch.ones_like(zs_pred_aux_maknee, device=maybe_gpu))
                    loss_discr_aux_1 = loss_discr_aux_1 / 2 * COEFF_DISCR
                    metrics_acc['batchw']['loss_discr_aux_1'].append(
                        loss_discr_aux_1.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_discr_aux_1'][-1]

                    # ---------------- Validate segmentation network ------------
                    # With source
                    ys_pred_out_oai, ys_pred_aux_oai = self.models['segm'](xs_oai)

                    # Output representations
                    loss_segm_out = self.losses['segm'](input_=ys_pred_out_oai,
                                                        target=ys_true_arg_oai)
                    loss_segm_out = loss_segm_out * COEFF_SEGM_OUT
                    metrics_acc['batchw']['loss_segm_out'].append(loss_segm_out.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_segm_out'][-1]

                    # Penultimate representations
                    loss_segm_aux = self.losses['segm'](input_=ys_pred_aux_oai,
                                                        target=ys_true_arg_oai)
                    loss_segm_aux = loss_segm_aux * COEFF_SEGM_AUX
                    metrics_acc['batchw']['loss_segm_aux'].append(loss_segm_aux.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_segm_aux'][-1]

                    # With target
                    ys_pred_out_maknee, ys_pred_aux_maknee = self.models['segm'](xs_maknee)

                    ys_pred_out_softmax_maknee = torch_fn.softmax(ys_pred_out_maknee, dim=1)
                    ys_pred_aux_softmax_maknee = torch_fn.softmax(ys_pred_aux_maknee, dim=1)
                    zs_pred_out_maknee = self.models['discr_out'](ys_pred_out_softmax_maknee)
                    zs_pred_aux_maknee = self.models['discr_aux'](ys_pred_aux_softmax_maknee)

                    # Use 0 as a label for the source domain
                    # Output representations
                    loss_advers_out = self.losses['advers'](
                        input=zs_pred_out_maknee,
                        target=torch.zeros_like(zs_pred_out_maknee, device=maybe_gpu))
                    loss_advers_out = loss_advers_out * COEFF_ADVERS_OUT
                    metrics_acc['batchw']['loss_advers_out'].append(
                        loss_advers_out.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_advers_out'][-1]

                    # Penultimate representations
                    loss_advers_aux = self.losses['advers'](
                        input=zs_pred_aux_maknee,
                        target=torch.zeros_like(zs_pred_aux_maknee, device=maybe_gpu))
                    loss_advers_aux = loss_advers_aux * COEFF_ADVERS_AUX
                    metrics_acc['batchw']['loss_advers_aux'].append(
                        loss_advers_aux.item())
                    metrics_acc['batchw']['loss_total'][-1] += \
                        metrics_acc['batchw']['loss_advers_aux'][-1]

                    if step_idx % 10 == 0:
                        self.tensorboard.add_scalars(
                            f'fold_{self.fold_idx}/losses_val',
                            {'discr_0_aux_batchw':
                                 metrics_acc['batchw']['loss_discr_aux_0'][-1],
                             'discr_0_out_batchw':
                                 metrics_acc['batchw']['loss_discr_out_0'][-1],
                             'discr_1_aux_batchw':
                                 metrics_acc['batchw']['loss_discr_aux_1'][-1],
                             'discr_1_out_batchw':
                                 metrics_acc['batchw']['loss_discr_out_1'][-1],
                             'discr_sum_aux_batchw':
                                 (metrics_acc['batchw']['loss_discr_aux_0'][-1] +
                                  metrics_acc['batchw']['loss_discr_aux_1'][-1]),
                             'discr_sum_out_batchw':
                                 (metrics_acc['batchw']['loss_discr_out_0'][-1] +
                                  metrics_acc['batchw']['loss_discr_out_1'][-1]),
                             'discr_sum_all_batchw':
                                 (metrics_acc['batchw']['loss_discr_aux_0'][-1] +
                                  metrics_acc['batchw']['loss_discr_out_0'][-1] +
                                  metrics_acc['batchw']['loss_discr_aux_1'][-1] +
                                  metrics_acc['batchw']['loss_discr_out_1'][-1]),
                             'segm_aux_batchw': metrics_acc['batchw']['loss_segm_aux'][-1],
                             'segm_out_batchw': metrics_acc['batchw']['loss_segm_out'][-1],
                             'advers_aux_batchw': metrics_acc['batchw']['loss_advers_aux'][-1],
                             'advers_out_batchw': metrics_acc['batchw']['loss_advers_out'][-1],
                             'total_batchw': metrics_acc['batchw']['loss_total'][-1],
                             }, global_step=(epoch_idx * steps_total + step_idx))

                    # ------------------ Calculate metrics -------------------
                    ys_pred_arg_np_oai = torch.argmax(ys_pred_out_softmax_oai, 1).to('cpu').numpy()
                    ys_true_arg_np_oai = ys_true_arg_oai.to('cpu').numpy()

                    metrics_acc['datasetw']['cm_oai'] += confusion_matrix(
                        ys_pred_arg_np_oai, ys_true_arg_np_oai,
                        self.config['output_channels'])

                    # Don't consider repeating entries for the metrics calculation
                    if step_idx < steps_okoa:
                        xs_okoa, ys_true_okoa = data_batch_okoa['xs'], data_batch_okoa['ys']
                        fnames_acc['okoa'].extend(data_batch_okoa['path_image'])
                        xs_okoa = xs_okoa.to(maybe_gpu)

                        ys_pred_okoa, _ = self.models['segm'](xs_okoa)

                        ys_true_arg_okoa = torch.argmax(ys_true_okoa.long().to(maybe_gpu), dim=1)
                        ys_pred_softmax_okoa = torch_fn.softmax(ys_pred_okoa, dim=1)

                        ys_pred_arg_np_okoa = torch.argmax(ys_pred_softmax_okoa, 1).to('cpu').numpy()
                        ys_true_arg_np_okoa = ys_true_arg_okoa.to('cpu').numpy()

                        metrics_acc['datasetw']['cm_okoa'] += confusion_matrix(
                            ys_pred_arg_np_okoa, ys_true_arg_np_okoa,
                            self.config['output_channels'])

                    prog_bar.update(1)

            del [loader_oai_iter_old, loader_okoa_iter_old, loader_maknee_iter_old]
            gc.collect()

        for k, v in metrics_acc['samplew'].items():
            metrics_acc['samplew'][k] = np.asarray(v)
        metrics_acc['datasetw']['dice_score_oai'] = np.asarray(
            dice_score_from_cm(metrics_acc['datasetw']['cm_oai']))
        metrics_acc['datasetw']['dice_score_okoa'] = np.asarray(
            dice_score_from_cm(metrics_acc['datasetw']['cm_okoa']))
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
            loss_curr = metrics_val['datasetw']['loss_total']
            # Alternative validation criterion:
            # loss_curr = metrics_val['datasetw']['loss_segm'] + metrics_val['datasetw']['loss_advers']
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
                self.handlers_ckpt['discr_aux'].save_new_ckpt(
                    model=self.models['discr_aux'],
                    model_name=self.config['model_discr_aux'],
                    fold_idx=self.fold_idx,
                    epoch_idx=epoch_idx)
                self.handlers_ckpt['discr_out'].save_new_ckpt(
                    model=self.models['discr_out'],
                    model_name=self.config['model_discr_out'],
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
@click.option('--model_discr_out', default='discriminator_a')
@click.option('--model_discr_aux', default='discriminator_a')
@click.option('--pretrained', is_flag=True)
@click.option('--path_pretrained_segm', type=str, help='Path to .pth file')
@click.option('--restore_weights', is_flag=True)
@click.option('--input_channels', default=1, type=int)
@click.option('--output_channels', default=1, type=int)
@click.option('--mask_mode', default='', type=str)
@click.option('--sample_mode', default='x_y', type=str)
@click.option('--loss_segm', default='multi_ce_loss')
@click.option('--lr_segm', default=0.0001, type=float)
@click.option('--lr_discr', default=0.0001, type=float)
@click.option('--optimizer_segm', default='adam')
@click.option('--optimizer_discr', default='adam')
@click.option('--batch_size', default=64, type=int)
@click.option('--epoch_size', default=1.0, type=float)
@click.option('--epoch_num', default=2, type=int)
@click.option('--fold_num', default=5, type=int)
@click.option('--fold_idx', default=-1, type=int)
@click.option('--fold_idx_ignore', multiple=True, type=int)
@click.option('--num_workers', default=1, type=int)
@click.option('--seed_trainval_test', default=0, type=int)
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
                                selection=('oai_imo', 'okoa', 'maknee'),
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
    folds = list(zip(sources['oai_imo']['trainval_folds'],
                     sources['okoa']['trainval_folds'],
                     sources['maknee']['trainval_folds']))

    for fold_idx, idcs_subsets in enumerate(folds):
        if fold_idx not in fold_idcs:
            continue
        logger.info(f'Training fold {fold_idx}')

        (sources['oai_imo']['train_idcs'], sources['oai_imo']['val_idcs']) = idcs_subsets[0]
        (sources['okoa']['train_idcs'], sources['okoa']['val_idcs']) = idcs_subsets[1]
        (sources['maknee']['train_idcs'], sources['maknee']['val_idcs']) = idcs_subsets[2]

        sources['oai_imo']['train_df'] = sources['oai_imo']['trainval_df'].iloc[sources['oai_imo']['train_idcs']]
        sources['oai_imo']['val_df'] = sources['oai_imo']['trainval_df'].iloc[sources['oai_imo']['val_idcs']]
        sources['okoa']['train_df'] = sources['okoa']['trainval_df'].iloc[sources['okoa']['train_idcs']]
        sources['okoa']['val_df'] = sources['okoa']['trainval_df'].iloc[sources['okoa']['val_idcs']]
        sources['maknee']['train_df'] = sources['maknee']['trainval_df'].iloc[sources['maknee']['train_idcs']]
        sources['maknee']['val_df'] = sources['maknee']['trainval_df'].iloc[sources['maknee']['val_idcs']]

        for n, s in sources.items():
            logger.info('Made {} train-val split, number of samples: {}, {}'
                        .format(n, len(s['train_df']), len(s['val_df'])))

        datasets = defaultdict(dict)

        datasets['oai_imo']['train'] = DatasetOAIiMoSagittal2d(
            df_meta=sources['oai_imo']['train_df'],
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
        datasets['okoa']['train'] = DatasetOKOASagittal2d(
            df_meta=sources['okoa']['train_df'],
            mask_mode='background_femoral_unitibial',
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
        datasets['maknee']['train'] = DatasetMAKNEESagittal2d(
            df_meta=sources['maknee']['train_df'],
            mask_mode='',
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
        datasets['oai_imo']['val'] = DatasetOAIiMoSagittal2d(
            df_meta=sources['oai_imo']['val_df'],
            mask_mode=config['mask_mode'],
            sample_mode=config['sample_mode'],
            transforms=[
                PercentileClippingAndToFloat(cut_min=10, cut_max=99),
                CenterCrop(height=300, width=300),
                Normalize(mean=0.252699, std=0.251142),
                ToTensor()
            ])
        datasets['okoa']['val'] = DatasetOKOASagittal2d(
            df_meta=sources['okoa']['val_df'],
            mask_mode='background_femoral_unitibial',
            sample_mode=config['sample_mode'],
            transforms=[
                PercentileClippingAndToFloat(cut_min=10, cut_max=99),
                CenterCrop(height=300, width=300),
                Normalize(mean=0.252699, std=0.251142),
                ToTensor()
            ])
        datasets['maknee']['val'] = DatasetMAKNEESagittal2d(
            df_meta=sources['maknee']['val_df'],
            mask_mode='',
            sample_mode=config['sample_mode'],
            transforms=[
                PercentileClippingAndToFloat(cut_min=10, cut_max=99),
                CenterCrop(height=300, width=300),
                Normalize(mean=0.252699, std=0.251142),
                ToTensor()
            ])

        loaders = defaultdict(dict)

        loaders['oai_imo']['train'] = DataLoader(
            datasets['oai_imo']['train'],
            batch_size=int(config['batch_size'] / 2),
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True)
        loaders['oai_imo']['val'] = DataLoader(
            datasets['oai_imo']['val'],
            batch_size=int(config['batch_size'] / 2),
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=True)
        loaders['okoa']['train'] = DataLoader(
            datasets['okoa']['train'],
            batch_size=int(config['batch_size'] / 2),
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True)
        loaders['okoa']['val'] = DataLoader(
            datasets['okoa']['val'],
            batch_size=int(config['batch_size'] / 2),
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=True)
        loaders['maknee']['train'] = DataLoader(
            datasets['maknee']['train'],
            batch_size=int(config['batch_size'] / 2),
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True)
        loaders['maknee']['val'] = DataLoader(
            datasets['maknee']['val'],
            batch_size=int(config['batch_size'] / 2),
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=True)

        trainer = ModelTrainer(config=config, fold_idx=fold_idx)

        tmp = trainer.fit(loaders=loaders)
        metrics_train, fnames_train, metrics_val, fnames_val = tmp

        fold_scores[fold_idx] = (metrics_val['datasetw']['dice_score_oai'],
                                 metrics_val['datasetw']['dice_score_okoa'])
        trainer.tensorboard.close()
    logger.info(f'Fold scores:\n{repr(fold_scores)}')


if __name__ == '__main__':
    main()

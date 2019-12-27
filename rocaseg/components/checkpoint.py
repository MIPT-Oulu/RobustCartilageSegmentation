import os
from glob import glob
import logging

import torch


logging.basicConfig()
logger = logging.getLogger('handler')
logger.setLevel(logging.DEBUG)


class CheckpointHandler(object):
    def __init__(self, path_root,
                 fname_pattern=('{model_name}__'
                                'fold_{fold_idx}__'
                                'epoch_{epoch_idx:>03d}.pth'),
                 num_saved=2):
        self.path_root = path_root
        self.fname_pattern = fname_pattern
        self.num_saved = num_saved

        ext = self.fname_pattern.split('.')[-1]

        if not os.path.exists(path_root):
            raise ValueError(f'Path {path_root} does not exist')

        full_pattern = os.path.join(self.path_root, '*.' + ext)
        self._all_ckpts = sorted(glob(full_pattern, recursive=False))
        logger.info(f'Checkpoints found: {len(self._all_ckpts)}')

        self._remove_excessive_ckpts()

    def _remove_excessive_ckpts(self):
        while len(self._all_ckpts) > self.num_saved:
            try:
                os.remove(self._all_ckpts[0])
                logger.info(f'Removed ckpt: {self._all_ckpts[0]}')
                self._all_ckpts = self._all_ckpts[1:]
            except OSError:
                logger.error(f'Cannot remove {self._all_ckpts[0]}')

    def get_last_ckpt(self):
        if len(self._all_ckpts) == 0:
            logger.warning(f'No checkpoints are available in {self.path_root}')
            return None
        else:
            fname_ckpt_sel = self._all_ckpts[-1]
            return fname_ckpt_sel

    def save_new_ckpt(self, model, model_name, fold_idx, epoch_idx):
        fname = self.fname_pattern.format(model_name=model_name,
                                          fold_idx=fold_idx,
                                          epoch_idx=epoch_idx)
        path_full = os.path.join(self.path_root, fname)
        try:
            torch.save(model.module.state_dict(), path_full)
        except AttributeError:
            torch.save(model.state_dict(), path_full)

        self._all_ckpts.append(path_full)
        self._remove_excessive_ckpts()

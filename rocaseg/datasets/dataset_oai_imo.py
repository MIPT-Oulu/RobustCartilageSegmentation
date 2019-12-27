import os
import glob
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import cv2
from torch.utils.data.dataset import Dataset

from rocaseg.datasets.constants import locations_mh53


logging.basicConfig()
logger = logging.getLogger('dataset')
logger.setLevel(logging.DEBUG)


def index_from_path_oai_imo(path_root, force=False):
    fname_meta_dyn = os.path.join(path_root, 'meta_dynamic.csv')
    fname_meta_base = os.path.join(path_root, 'meta_base.csv')

    if not os.path.exists(fname_meta_dyn) or force:
        fnames_image = glob.glob(
            os.path.join(path_root, '**', 'images', '*.png'), recursive=True)
        logger.info('{} images found'.format(len(fnames_image)))
        fnames_mask = glob.glob(
            os.path.join(path_root, '**', 'masks', '*.png'), recursive=True)
        logger.info('{} masks found'.format(len(fnames_mask)))

        df_meta = pd.read_csv(fname_meta_base,
                              dtype={'patient': str,
                                     'release': str,
                                     'prefix_var': str,
                                     'sequence': str,
                                     'side': str,
                                     'slice_idx': int,
                                     'pixel_spacing_0': float,
                                     'pixel_spacing_1': float,
                                     'slice_thickness': float,
                                     'KL': int},
                              index_col=False)

        if len(fnames_image) != len(df_meta):
            raise ValueError("Number of images doesn't match with the metadata")
        if len(fnames_mask) != len(df_meta):
            raise ValueError("Number of masks doesn't match with the metadata")

        df_meta['path_image'] = [os.path.join(path_root, e)
                                 for e in df_meta['path_rel_image']]
        df_meta['path_mask'] = [os.path.join(path_root, e)
                                for e in df_meta['path_rel_mask']]

        # Check for in-slice mask presence
        def worker_nv7(fname):
            mask = read_mask(path_file=fname, mask_mode='raw')
            if np.any(mask[1:] > 0):
                return 1
            else:
                return 0

        logger.info('Exploring the annotations')
        tmp = Parallel(n_jobs=-1)(
            delayed(worker_nv7)(row['path_mask'])
            for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)))
        df_meta['has_mask'] = tmp

        # Sort the records
        df_meta_sorted = (df_meta
                          .sort_values(['patient', 'prefix_var',
                                        'sequence', 'slice_idx'])
                          .reset_index()
                          .drop('index', axis=1))

        df_meta_sorted.to_csv(fname_meta_dyn, index=False)
    else:
        df_meta_sorted = pd.read_csv(fname_meta_dyn,
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
    return df_meta_sorted


def read_image(path_file):
    image = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)
    return image.reshape((1, *image.shape))


def read_mask(path_file, mask_mode):
    """Read mask from the file, and pre-process it.

    IMPORTANT: currently, we handle the inter-class collisions by assigning
    the joint pixels to a class with a lower index.

    Parameters
    ----------
    path_file: str
        Full path to mask file.
    mask_mode: str
        Specifies which channels of mask to use.

    Returns
    -------
    out : (ch, d0, d1) uint8 ndarray

    """
    mask = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)

    locations = {
        '_background': (locations_mh53['Background'], ),
        'femoral': (locations_mh53['FemoralCartilage'], ),
        'tibial': (locations_mh53['LateralTibialCartilage'],
                   locations_mh53['MedialTibialCartilage']),
        'patellar': (locations_mh53['PatellarCartilage'], ),
        'menisci': (locations_mh53['LateralMeniscus'],
                    locations_mh53['MedialMeniscus']),
    }

    if mask_mode == 'raw':
        return mask
    elif mask_mode == 'all_unitibial_unimeniscus':
        ret = np.empty((5, *mask.shape), dtype=mask.dtype)
        ret[0, :, :] = np.isin(mask, locations['_background']).astype(np.uint8)
        ret[1, :, :] = np.isin(mask, locations['femoral']).astype(np.uint8)
        ret[2, :, :] = np.isin(mask, locations['tibial']).astype(np.uint8)
        ret[3, :, :] = np.isin(mask, locations['patellar']).astype(np.uint8)
        ret[4, :, :] = np.isin(mask, locations['menisci']).astype(np.uint8)
        return ret
    else:
        raise ValueError('Invalid `mask_mode`')


class DatasetOAIiMoSagittal2d(Dataset):
    def __init__(self, df_meta, mask_mode=None, name=None, transforms=None,
                 sample_mode='x_y', **kwargs):
        logger.warning('Redundant dataset init arguments:\n{}'
                       .format(repr(kwargs)))

        self.df_meta = df_meta
        self.mask_mode = mask_mode
        self.name = name
        self.transforms = transforms
        self.sample_mode = sample_mode

    def __len__(self):
        return len(self.df_meta)

    def _getitem_x_y(self, idx):
        image = read_image(self.df_meta['path_image'].iloc[idx])
        mask = read_mask(self.df_meta['path_mask'].iloc[idx], self.mask_mode)

        # Apply transformations
        if self.transforms is not None:
            for t in self.transforms:
                if hasattr(t, 'randomize'):
                    t.randomize()
                image, mask = t(image, mask)

        tmp = dict(self.df_meta.iloc[idx])
        tmp['image'] = image
        tmp['mask'] = mask

        tmp['xs'] = tmp['image']
        tmp['ys'] = tmp['mask']
        return tmp

    def __getitem__(self, idx):
        if self.sample_mode == 'x_y':
            return self._getitem_x_y(idx)
        else:
            raise ValueError('Invalid `sample_mode`')

    def read_image(self, path_file):
        return read_image(path_file=path_file)

    def read_mask(self, path_file):
        return read_mask(path_file=path_file, mask_mode=self.mask_mode)

    def describe(self):
        summary = defaultdict(float)
        for i in range(len(self)):
            if self.sample_mode == 'x_y':
                _, mask = self.__getitem__(i)
            else:
                mask = self.__getitem__(i)['mask']
            summary['num_class_pixels'] += mask.numpy().sum(axis=(1, 2))
        summary['class_importance'] = \
            np.sum(summary['num_class_pixels']) / summary['num_class_pixels']
        summary['class_importance'] /= np.sum(summary['class_importance'])
        logger.info('Dataset statistics:')
        logger.info(sorted(summary.items()))

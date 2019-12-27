import os
import click
from tqdm import tqdm

import numpy as np
import pandas as pd
import cv2


@click.command()
@click.option('--path_root_in', help='E.g. data/31_OKOA_full_meta')
@click.option('--spacing_in', nargs=2, default=(0.5859375, 0.5859375))
@click.option('--path_root_out', help='E.g. data/32_OKOA_full_meta_rescaled')
@click.option('--spacing_out', nargs=2, default=(0.36458333, 0.36458333))
@click.option('--dirname_images', default='images')
@click.option('--dirname_masks', default='masks')
@click.option('--num_threads', default=12, type=click.IntRange(-1, 12))
@click.option('--margin', default=0, type=int)
@click.option('--update_meta', is_flag=True)
def main(**config):
    # Get the index of image files and the corresponding metadata
    path_meta = os.path.join(config['path_root_in'], 'meta_base.csv')
    if os.path.exists(path_meta):
        pass
    else:
        path_meta = os.path.join(config['path_root_in'], 'meta_dynamic.csv')

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

    df_in = df_meta.sort_values(['patient', 'release', 'sequence', 'side', 'slice_idx'])

    ratio = (np.asarray(config['spacing_in']) /
             np.asarray(config['spacing_out']))

    groupers_stack = ['patient', 'release', 'sequence', 'side', 'slice_idx']

    # Resample images
    if config['dirname_images'] is not None:
        for name_gb, df_gb in tqdm(df_in.groupby(groupers_stack), desc='Resample images'):
            patient, release, sequence, side, slice_idx = name_gb

            fn_base = f'{slice_idx:03d}.png'
            dir_in = os.path.join(config['path_root_in'],
                                  patient, release, sequence,
                                  config['dirname_images'])
            dir_out = os.path.join(config['path_root_out'],
                                   patient, release, sequence,
                                   config['dirname_images'])
            os.makedirs(dir_out, exist_ok=True)

            path_in = os.path.join(dir_in, fn_base)
            path_out = os.path.join(dir_out, fn_base)

            img_in = cv2.imread(path_in, cv2.IMREAD_GRAYSCALE)

            if config['margin'] == 0:
                tmp = img_in
            else:
                tmp = img_in[config['margin']:-config['margin'],
                             config['margin']:-config['margin']]

            shape_out = tuple(np.floor(tmp.shape * ratio).astype(np.int))[::-1]
            tmp = cv2.resize(tmp, shape_out)
            img_out = tmp

            cv2.imwrite(path_out, img_out)

    # Resample masks
    if config['dirname_masks'] is not None:
        for name_gb, df_gb in tqdm(df_in.groupby(groupers_stack), desc='Resample masks'):
            patient, release, sequence, side, slice_idx = name_gb

            fn_base = f'{slice_idx:03d}.png'
            dir_in = os.path.join(config['path_root_in'],
                                  patient, release, sequence,
                                  config['dirname_masks'])
            dir_out = os.path.join(config['path_root_out'],
                                   patient, release, sequence,
                                   config['dirname_masks'])
            os.makedirs(dir_out, exist_ok=True)

            path_in = os.path.join(dir_in, fn_base)
            if not os.path.exists(path_in):
                print(f'No mask found for {name_gb}')
                continue
            path_out = os.path.join(dir_out, fn_base)

            mask_in = cv2.imread(path_in, cv2.IMREAD_GRAYSCALE)

            if config['margin'] == 0:
                tmp = mask_in
            else:
                tmp = mask_in[config['margin']:-config['margin'],
                              config['margin']:-config['margin']]

            shape_out = tuple(np.floor(tmp.shape * ratio).astype(np.int))[::-1]
            tmp = cv2.resize(tmp, shape_out, interpolation=cv2.INTER_NEAREST)
            mask_out = tmp

            cv2.imwrite(path_out, mask_out)

    if config['update_meta']:
        df_out = (df_in.assign(pixel_spacing_0=config['spacing_out'][0])
                       .assign(pixel_spacing_1=config['spacing_out'][1]))
        df_out.to_csv(os.path.join(config['path_root_out'], 'meta_base.csv'), index=False)


if __name__ == '__main__':
    main()

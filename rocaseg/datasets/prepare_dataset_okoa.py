import os
from collections import defaultdict
import click

from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

import pydicom
import cv2


cv2.ocl.setUseOpenCL(False)


def read_dicom(fname, only_data=False):
    data = pydicom.read_file(fname)
    if len(data.PixelData) == 131072:
        dtype = np.uint16
    else:
        dtype = np.uint8
    image = np.frombuffer(data.PixelData, dtype=dtype).astype(float)

    if data.PhotometricInterpretation == 'MONOCHROME1':
        image = image.max() - image

    image = image.reshape((data.Rows, data.Columns))

    if only_data:
        return image
    else:
        if hasattr(data, 'ImagerPixelSpacing'):
            spacing = [float(e) for e in data.ImagerPixelSpacing]
            slice_thickness = float(data.SliceThickness)
        elif hasattr(data, 'PixelSpacing'):
            spacing = [float(e) for e in data.PixelSpacing]
            slice_thickness = float(data.SliceThickness)
        else:
            msg = f'DICOM {fname} does not have the required attributes'
            print(msg)
            spacing = (0.0, 0.0)
            slice_thickness = 0.0
        return image, spacing[0], spacing[1], slice_thickness


@click.command()
@click.argument('path_root_okoa')
@click.argument('path_root_output')
@click.option('--num_threads', default=12, type=click.IntRange(0, 16))
@click.option('--margin', default=0, type=int)
@click.option('--meta_only', is_flag=True)
def main(**config):
    config['path_root_okoa'] = os.path.abspath(config['path_root_okoa'])
    config['path_root_output'] = os.path.abspath(config['path_root_output'])

    # -------------------------------------------------------------------------
    def worker_s5g(path_root_output, row, margin):
        meta = defaultdict(list)

        patient = row['patient'].values[0]
        slice_idx = row['slice_idx'].values[0]
        side = row['side'].values[0]
        release = 'initial'
        sequence = 't2_de3d_we_sag_iso'

        image, *dicom_meta = read_dicom(row[('fname_full', 'Images')])

        # Set the default values for the voxel spacings
        pixel_spacing_0 = dicom_meta[0] or '0.5859375'
        pixel_spacing_1 = dicom_meta[1] or '0.5859375'
        slice_thickness = dicom_meta[2] or '0.60000002384186'

        mask_femur = read_dicom(row[('fname_full', 'Femur')], only_data=True)
        mask_tibia = read_dicom(row[('fname_full', 'Tibia')], only_data=True)
        mask_full = np.zeros(mask_femur.shape, dtype=np.uint8)
        # Use inverse order to prioritize femoral tissues in collision handling
        mask_full[mask_tibia > 0] = 2
        mask_full[mask_femur > 0] = 1

        if margin:
            image = image[margin:-margin, margin:-margin]
            mask_full = mask_full[margin:-margin, margin:-margin]

        fname_pattern = '{slice_idx:>03}.{ext}'

        # Save image and mask data
        dir_rel_image = os.path.join(patient, release, sequence, 'images')
        dir_rel_mask = os.path.join(patient, release, sequence, 'masks')
        dir_abs_image = os.path.join(path_root_output, dir_rel_image)
        dir_abs_mask = os.path.join(path_root_output, dir_rel_mask)
        for d in (dir_abs_image, dir_abs_mask):
            if not os.path.exists(d):
                os.makedirs(d)

        fname_image = fname_pattern.format(slice_idx=slice_idx, ext='png')
        path_abs_image = os.path.join(dir_abs_image, fname_image)
        if not config['meta_only']:
            cv2.imwrite(path_abs_image, image)

        fname_mask = fname_pattern.format(slice_idx=slice_idx, ext='png')
        path_abs_mask = os.path.join(dir_abs_mask, fname_mask)
        if not config['meta_only']:
            cv2.imwrite(path_abs_mask, mask_full)

        path_rel_image = os.path.join(dir_rel_image, fname_image)
        path_rel_mask = os.path.join(dir_rel_mask, fname_mask)

        meta['subset'].append(row['subset'].values[0])
        meta['patient'].append(patient)
        meta['release'].append(release)
        meta['sequence'].append(sequence)
        meta['side'].append(side)
        meta['KL'].append(row['KL'].values[0])
        meta['slice_idx'].append(slice_idx)
        meta['pixel_spacing_0'].append(pixel_spacing_0)
        meta['pixel_spacing_1'].append(pixel_spacing_1)
        meta['slice_thickness'].append(slice_thickness)
        meta['path_rel_image'].append(path_rel_image)
        meta['path_rel_mask'].append(path_rel_mask)
        return meta

    # -------------------------------------------------------------------------

    # Get list of images files
    paths_fnames_dicom = glob(os.path.join(config['path_root_okoa'], '**', '*.IMA'),
                              recursive=True)

    # root / training|evaluation / P36 / Images|Femur|Tibia / (1-160).IMA
    def meta_from_fname(fn):
        tmp = fn.split('/')
        slice_idx = int(os.path.splitext(tmp[-1])[0]) - 1
        slice_idx = '{:>03}'.format(slice_idx)
        meta = {
            'fname_full': fn,
            'slice_idx': slice_idx,
            'kind': tmp[-2],
            'patient': tmp[-3],
            'subset': tmp[-4]
        }
        return meta

    dict_meta = {
        'fname_full': [],
        'slice_idx': [],
        'kind': [],
        'patient': [],
        'subset': []
    }

    for e in paths_fnames_dicom:
        tmp_meta = meta_from_fname(e)
        for k, v in tmp_meta.items():
            dict_meta[k].append(v)

    df_meta = pd.DataFrame.from_dict(dict_meta)
    df_meta = (df_meta
               .set_index(['subset', 'patient', 'slice_idx', 'kind'])
               .unstack('kind')
               .reset_index())

    # Add info on scan side and KL grades
    path_file_side = os.path.join(config['path_root_okoa'], 'sides.csv')
    df_side = pd.read_csv(path_file_side)
    path_file_kl = os.path.join(config['path_root_okoa'], 'KL_grades.csv')
    df_kl = pd.read_csv(path_file_kl)

    df_extra = pd.merge(df_side, df_kl, on='patient', how='left', sort=True)
    df_extra.loc[:, 'KL'] = -1
    for r_idx, r in df_extra.iterrows():
        if r['side'] == 'LEFT':
            df_extra.loc[r_idx, 'KL'] = r['KL left']
        elif r['side'] == 'RIGHT':
            df_extra.loc[r_idx, 'KL'] = r['KL right']

    # Keep only the fields of interest
    df_extra = df_extra.loc[:, ['patient', 'side', 'KL', 'age']]

    # Make same multi-index such than pd.merge doesn't create extra column
    df_extra.columns = pd.MultiIndex.from_tuples([(c, '') for c in df_extra.columns])

    # Merge the metadata into the single df
    df_meta = pd.merge(df_meta, df_extra, on='patient', how='left', sort=True)

    # Process the raw data
    metas = Parallel(config['num_threads'])(delayed(worker_s5g)(
        *[config['path_root_output'], row, config['margin']]
    ) for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)))

    # Merge meta information from different stacks
    tmp = defaultdict(list)
    for d in metas:
        for k, v in d.items():
            tmp[k].extend(v)
    df_out = pd.DataFrame.from_dict(tmp)

    path_output_meta = os.path.join(config['path_root_output'], 'meta_base.csv')
    df_out.to_csv(path_output_meta, index=False)


if __name__ == '__main__':
    main()

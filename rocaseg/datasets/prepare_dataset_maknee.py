import os
from collections import defaultdict
from glob import glob

import click
from joblib import Parallel, delayed
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
            spacing = [float(e) for e in data.ImagerPixelSpacing[:2]]
            slice_thickness = float(data.SliceThickness)
        elif hasattr(data, 'PixelSpacing'):
            spacing = [float(e) for e in data.PixelSpacing[:2]]
            slice_thickness = float(data.SliceThickness)
        else:
            msg = f'DICOM {fname} does not contain spacing info'
            print(msg)
            spacing = (0.0, 0.0)
            slice_thickness = 0.0

        if data.Laterality == 'R':
            side = 'RIGHT'
        elif data.Laterality == 'L':
            side = 'LEFT'
        else:
            msg = 'DICOM {fname} does not contain side info'
            raise AttributeError(msg)
        return image, spacing[0], spacing[1], slice_thickness, side


@click.command()
@click.argument('path_root_maknee')
@click.argument('path_root_output')
@click.option('--num_threads', default=12, type=click.IntRange(0, 16))
@click.option('--margin', default=0, type=int)
@click.option('--meta_only', is_flag=True)
def main(**config):
    config['path_root_maknee'] = os.path.abspath(config['path_root_maknee'])
    config['path_root_output'] = os.path.abspath(config['path_root_output'])

    # -------------------------------------------------------------------------
    def worker(path_root_output, row, margin):
        meta = defaultdict(list)

        patient = row['patient']
        slice_idx = row['slice_idx']
        release = 'initial'
        sequence = 't2_de3d_we_sag_iso'

        image, *dicom_meta = read_dicom(row['fname_full_image'])

        side = dicom_meta[3]

        if margin != 0:
            image = image[margin:-margin, margin:-margin]

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

        path_rel_image = os.path.join(dir_rel_image, fname_image)

        meta['patient'].append(patient)
        meta['release'].append(release)
        meta['sequence'].append(sequence)
        meta['side'].append(side)
        meta['slice_idx'].append(slice_idx)
        meta['pixel_spacing_0'].append(dicom_meta[0])
        meta['pixel_spacing_1'].append(dicom_meta[1])
        meta['slice_thickness'].append(dicom_meta[2])
        meta['path_rel_image'].append(path_rel_image)
        return meta

    # -------------------------------------------------------------------------

    # Get list of images files
    fnames_dicom = glob(os.path.join(config['path_root_maknee'],
                                     'MRI', 'Scans', '**',
                                     't2_de3d_we_sag_iso*', 'IMG*'),
                        recursive=True)
    fnames_dicom = list(sorted(fnames_dicom))

    def meta_from_fname(fn):
        # root / MRI / Scans / 001 / t2_de3d_we_sag_iso / IMG00000
        tmp = fn.split('/')
        meta = {
            'fname_full_image': fn,
            'slice_idx': os.path.splitext(tmp[-1])[0][-3:],
            'patient': 'P{:>03}'.format(tmp[-3])}
        return meta

    dict_meta = {
        'fname_full_image': [],
        'slice_idx': [],
        'patient': []}

    for e in fnames_dicom:
        tmp_meta = meta_from_fname(e)
        for k, v in tmp_meta.items():
            dict_meta[k].append(v)

    df_meta = pd.DataFrame.from_dict(dict_meta)

    metas = Parallel(config['num_threads'])(delayed(worker)(
        *[config['path_root_output'], row, config['margin']]
    ) for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)))

    # Merge meta information from different stacks
    tmp = defaultdict(list)
    for d in metas:
        for k, v in d.items():
            tmp[k].extend(v)
    df_meta = pd.DataFrame.from_dict(tmp)

    # Add grading data to the meta-info df
    path_file_exp = os.path.join(config['path_root_maknee'], 'MAKnee_KL_subjects.xlsx')
    df_kl = pd.read_excel(path_file_exp)

    df_meta_uniq = df_meta.loc[:, ['patient', 'side']].drop_duplicates()
    df_kl.loc[:, 'ID'] = ['P{:>03}'.format(e) for e in df_kl['ID']]
    df_kl = df_kl.set_index(df_kl['ID'])
    tmp_kl = []

    for _, row in df_meta_uniq.iterrows():
        tmp_patient = row['patient']
        tmp_side = row['side']

        if tmp_side == 'RIGHT':
            tmp_kl.append(int(df_kl.loc[tmp_patient, 'KL right']))
        elif tmp_side == 'LEFT':
            tmp_kl.append(int(df_kl.loc[tmp_patient, 'KL left']))
        else:
            msg = f'Unexpected side value {tmp_side}'
            raise ValueError(msg)

    df_meta_uniq['KL'] = tmp_kl
    df_meta = pd.merge(df_meta, df_meta_uniq, on=['patient', 'side'], how='left')

    df_out = df_meta

    path_output_meta = os.path.join(config['path_root_output'], 'meta_base.csv')
    df_out.to_csv(path_output_meta, index=False)


if __name__ == '__main__':
    main()

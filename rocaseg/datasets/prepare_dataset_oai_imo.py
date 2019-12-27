import os
from glob import glob
from collections import defaultdict

import click
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
from sas7bdat import SAS7BDAT
from scipy import io
import pandas as pd

import pydicom
import cv2

from rocaseg.datasets.constants import locations_mh53
from rocaseg.datasets.meta_oai import side_code_to_str, release_to_prefix_var


cv2.ocl.setUseOpenCL(False)


def read_dicom(fname):
    data = pydicom.read_file(fname)
    image = np.frombuffer(data.PixelData, dtype=np.uint16).astype(float)

    if data.PhotometricInterpretation == 'MONOCHROME1':
        image = image.max() - image
    image = image.reshape((data.Rows, data.Columns))

    if 'RIGHT' in data.SeriesDescription:
        side = 'RIGHT'
    elif 'LEFT' in data.SeriesDescription:
        side = 'LEFT'
    else:
        print(data)
        msg = f'DICOM {fname} does not contain side info'
        raise ValueError(msg)

    if hasattr(data, 'ImagerPixelSpacing'):
        spacing = [float(e) for e in data.ImagerPixelSpacing[:2]]
    elif hasattr(data, 'PixelSpacing'):
        spacing = [float(e) for e in data.PixelSpacing[:2]]
    else:
        msg = f'DICOM {fname} does not contain spacing info'
        raise AttributeError(msg)

    return (image,
            spacing[0],
            spacing[1],
            float(data.SliceThickness),
            side)


def mask_from_mat(masks_mat, mask_shape, slice_idx, attr_name):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    data = getattr(masks_mat[0][slice_idx], attr_name)
    if len(data.shape) > 0:
        for comp in range(data.shape[1]):
            cnt = data[0, comp][:, :2].copy()
            cnt[:, 1] = mask_shape[0] - cnt[:, 1]
            cntf = cnt.astype(np.int)
            cv2.drawContours(mask, [cntf], -1, (255, 255, 255), -1)
    mask = (mask > 0).astype(np.uint8)
    return mask


@click.command()
@click.argument('path_root_oai_mri')
@click.argument('path_root_imo')
@click.argument('path_root_output')
@click.option('--num_threads', default=12, type=click.IntRange(0, 16))
@click.option('--margin', default=0, type=int)
@click.option('--meta_only', is_flag=True)
def main(**config):
    config['path_root_oai_mri'] = os.path.abspath(config['path_root_oai_mri'])
    config['path_root_imo'] = os.path.abspath(config['path_root_imo'])
    config['path_root_output'] = os.path.abspath(config['path_root_output'])

    # -------------------------------------------------------------------------
    def worker_xz9(path_root_output, path_stack, margin):
        meta = defaultdict(list)

        release, patient = path_stack.split('/')[-4:-2]
        prefix_var = release_to_prefix_var[release]
        sequence = 'sag_3d_dess_we'

        path_annot = os.path.join(config['path_root_imo'], patient, prefix_var)
        fnames_annot = glob(os.path.join(path_annot, '*.mat'))
        if len(fnames_annot) != 1:
            raise ValueError(f'Unexpected annotations for patient: {patient}')
        fname_annot = fnames_annot[0]

        file_mat = io.loadmat(os.path.join(path_annot, fname_annot),
                              struct_as_record=False)
        masks_mat = file_mat['datastruct']
        num_slices = masks_mat.shape[1]

        for slice_idx in range(num_slices):
            # Indexing of slices in OAI dataset starts with 001
            fname_src = os.path.join(path_stack, '{:>03}'.format(slice_idx+1))
            image, *dicom_meta = read_dicom(fname_src)

            side = dicom_meta[3]

            mask_proc = np.zeros_like(image)

            # NOTICE: Reference masks have some collisions. We solve them
            #         by prioritising the tissues which are earlier in the list.
            for part_name, part_value in reversed(locations_mh53.items()):
                # Skip the background as it is not presented in the source data
                if part_name == 'Background':
                    continue
                try:
                    mask_temp = mask_from_mat(masks_mat, image.shape,
                                              slice_idx, part_name)
                    mask_proc[mask_temp > 0] = part_value
                except AttributeError:
                    print(f'Error accessing {part_name} in {fname_src}')

            if margin != 0:
                image = image[margin:-margin, margin:-margin]
                mask_proc = mask_proc[margin:-margin, margin:-margin]

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
                cv2.imwrite(path_abs_mask, mask_proc)

            path_rel_image = os.path.join(dir_rel_image, fname_image)
            path_rel_mask = os.path.join(dir_rel_mask, fname_mask)

            meta['patient'].append(patient)
            meta['release'].append(release)
            meta['prefix_var'].append(prefix_var)
            meta['sequence'].append(sequence)
            meta['side'].append(side)
            meta['slice_idx'].append(slice_idx)
            meta['pixel_spacing_0'].append(dicom_meta[0])
            meta['pixel_spacing_1'].append(dicom_meta[1])
            meta['slice_thickness'].append(dicom_meta[2])
            meta['path_rel_image'].append(path_rel_image)
            meta['path_rel_mask'].append(path_rel_mask)
        return meta
    # -------------------------------------------------------------------------

    # OAI data path structure:
    #   root / examination / release / patient / date / barcode (/ slices)
    paths_stacks = glob(os.path.join(config['path_root_oai_mri'], '**/**/**/**/**'))
    paths_stacks.sort(key=lambda x: int(x.split('/')[-3]))

    metas = Parallel(config['num_threads'])(delayed(worker_xz9)(
        *[config['path_root_output'], path_stack, config['margin']]
    ) for path_stack in tqdm(paths_stacks))

    # Merge meta information from different stacks
    tmp = defaultdict(list)
    for d in metas:
        for k, v in d.items():
            tmp[k].extend(v)
    df_out = pd.DataFrame.from_dict(tmp)

    # Find the grading data
    fnames_sas = glob(os.path.join(config['path_root_oai_mri'],
                                   '*', '*.sas7bdat'), recursive=True)

    # Read semi-quantitative data
    dfs = dict()
    for fn in fnames_sas:
        with SAS7BDAT(fn) as f:
            raw = [r for r in f]
        tmp = pd.DataFrame(raw[1:], columns=raw[0])

        prefix_var = [c for c in tmp.columns if c.endswith('XRKL')][0][:3]

        tmp = tmp.rename(lambda x: x.upper(), axis=1)
        tmp = tmp.rename({'VERSION': f'{prefix_var}VERSION',
                          'ID': 'patient',
                          'SIDE': 'side'}, axis=1)

        tmp['side'] = tmp['side'].apply(lambda s: side_code_to_str[s])
        dfs.update({prefix_var: tmp})

    # Set the index to join on
    for k, tmp in dfs.items():
        dfs[k] = tmp.set_index(['patient', 'side', 'READPRJ'])

    df = pd.concat(dfs.values(), axis=1)
    df = df.reset_index()

    # Remove unnecessary columns and reformat the grading info
    df_sel = df[['patient', 'side', 'V00XRKL', 'V01XRKL']]
    df_sel = (df_sel
              .set_index(['patient', 'side'])
              .rename({'V00XRKL': 'V00', 'V01XRKL': 'V01'}, axis=1)
              .stack()
              .reset_index()
              .rename({'level_2': 'prefix_var', 0: 'KL'}, axis=1))

    # Select the subset for which the annotations are available
    indexers = ['patient', 'side', 'prefix_var']
    sel = df_out.set_index(indexers).index.unique()
    df_sel = (df_sel
              .drop_duplicates(subset=indexers)  # There are ~5 duplicates
              .set_index(indexers)
              .loc[sel, :]
              .reset_index())

    df_out = pd.merge(df_out, df_sel, on=indexers, how='left')

    path_output_meta = os.path.join(config['path_root_output'], 'meta_base.csv')
    df_out.to_csv(path_output_meta, index=False)


if __name__ == '__main__':
    main()

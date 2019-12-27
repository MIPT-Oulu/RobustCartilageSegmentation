import os
import logging

from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from rocaseg.datasets import (index_from_path_oai_imo,
                              index_from_path_okoa,
                              index_from_path_maknee)


logging.basicConfig()
logger = logging.getLogger('datasets')
logger.setLevel(logging.DEBUG)


def sources_from_path(path_data_root,
                      selection=None,
                      with_folds=False,
                      fold_num=5,
                      seed_trainval_test=0):
    """

    Args:
        path_data_root: str

        selection: iterable or str or None

        with_folds: bool
            Whether to split trainval subset into the folds.
        fold_num: int
            Number of folds.
        seed_trainval_test: int
            Random state for the trainval/test splitting.

    Returns:

    """
    if selection is None:
        selection = ('oai_imo', 'okoa', 'maknee')
    elif isinstance(selection, str):
        selection = (selection, )

    sources = dict()

    for name in selection:
        if name == 'oai_imo':
            logger.info('--- OAI iMorphics dataset ---')
            tmp = dict()
            tmp['path_root'] = os.path.join(path_data_root,
                                            '91_OAI_iMorphics_full_meta')

            if not os.path.exists(tmp['path_root']):
                logger.warning(f"Dataset {name} is not found in {tmp['path_root']}")
                continue

            tmp['full_df'] = index_from_path_oai_imo(tmp['path_root'])
            logger.info(f"Total number of samples: "
                        f"{len(tmp['full_df'])}")

            # Select the specific subset
            # Remove two series from the dataset as they are completely missing
            # information on patellar cartilage:
            #        /0.C.2/9674570/20040913/10699609/
            #        /1.C.2/9674570/20050829/10488714/
            tmp['sel_df'] = tmp['full_df'][tmp['full_df']['patient'] != '9674570']
            logger.info(f"Selected number of samples: "
                        f"{len(tmp['sel_df'])}")

            if with_folds:
                # Get trainval/test split
                tmp_groups = tmp['sel_df'].loc[:, 'patient'].values
                tmp_grades = tmp['sel_df'].loc[:, 'KL'].values

                tmp_gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                            random_state=seed_trainval_test)
                tmp_idcs_trainval, tmp_idcs_test = next(tmp_gss.split(X=tmp['sel_df'],
                                                                      y=tmp_grades,
                                                                      groups=tmp_groups))
                tmp['trainval_df'] = tmp['sel_df'].iloc[tmp_idcs_trainval]
                tmp['test_df'] = tmp['sel_df'].iloc[tmp_idcs_test]
                logger.info(f"Made trainval-test split, number of samples: "
                            f"{len(tmp['trainval_df'])}, "
                            f"{len(tmp['test_df'])}")

                # Make k folds
                tmp_gkf = GroupKFold(n_splits=fold_num)
                tmp_groups = tmp['trainval_df'].loc[:, 'patient'].values
                tmp_grades = tmp['trainval_df'].loc[:, 'KL'].values

                tmp['trainval_folds'] = tmp_gkf.split(X=tmp['trainval_df'],
                                                      y=tmp_grades, groups=tmp_groups)
            sources['oai_imo'] = tmp

        elif name == 'okoa':
            logger.info('--- OKOA dataset ---')
            tmp = dict()
            tmp['path_root'] = os.path.join(path_data_root,
                                            '32_OKOA_full_meta_rescaled')

            if not os.path.exists(tmp['path_root']):
                logger.warning(f"Dataset {name} is not found in {tmp['path_root']}")
                continue

            tmp['full_df'] = index_from_path_okoa(tmp['path_root'])
            logger.info(f"Total number of samples: "
                        f"{len(tmp['full_df'])}")

            # Select the specific subset
            tmp['sel_df'] = tmp['full_df']
            logger.info(f"Selected number of samples: "
                        f"{len(tmp['sel_df'])}")

            if with_folds:
                # Get trainval/test split
                tmp['trainval_df'] = tmp['sel_df'][tmp['sel_df']['subset'] == 'training']
                tmp['test_df'] = tmp['sel_df'][tmp['sel_df']['subset'] == 'evaluation']
                logger.info(f"Made trainval-test split, number of samples: "
                            f"{len(tmp['trainval_df'])}, "
                            f"{len(tmp['test_df'])}")

                # Make k folds
                tmp_gkf = GroupKFold(n_splits=fold_num)
                tmp_groups = tmp['trainval_df'].loc[:, 'patient'].values

                tmp['trainval_folds'] = tmp_gkf.split(X=tmp['trainval_df'],
                                                      groups=tmp_groups)
            sources['okoa'] = tmp

        elif name == 'maknee':
            logger.info('--- MAKNEE dataset ---')
            tmp = dict()
            tmp['path_root'] = os.path.join(path_data_root,
                                            '42_MAKNEE_full_meta_rescaled')

            if not os.path.exists(tmp['path_root']):
                logger.warning(f"Dataset {name} is not found in {tmp['path_root']}")
                continue

            tmp['full_df'] = index_from_path_maknee(tmp['path_root'])
            logger.info(f"Total number of samples: "
                        f"{len(tmp['full_df'])}")

            # Select the specific subset
            tmp['sel_df'] = tmp['full_df']
            logger.info(f"Selected number of samples: "
                        f"{len(tmp['sel_df'])}")

            # Get trainval/test split
            tmp_groups = tmp['sel_df'].loc[:, 'patient'].values

            tmp_gss = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                        random_state=seed_trainval_test)
            tmp_idcs_trainval, tmp_idcs_test = next(tmp_gss.split(X=tmp['sel_df'],
                                                                  groups=tmp_groups))
            tmp['trainval_df'] = tmp['sel_df'].iloc[tmp_idcs_trainval]
            tmp['test_df'] = tmp['sel_df'].iloc[tmp_idcs_test]
            logger.info(f"Made trainval-test split, number of samples: "
                        f"{len(tmp['trainval_df'])}, "
                        f"{len(tmp['test_df'])}")

            if with_folds:
                # Make k folds
                tmp_gkf = GroupKFold(n_splits=fold_num)
                tmp_groups = tmp['trainval_df'].loc[:, 'patient'].values

                tmp['trainval_folds'] = tmp_gkf.split(X=tmp['trainval_df'],
                                                      groups=tmp_groups)
            sources['maknee'] = tmp

        else:
            raise ValueError(f'Unknown dataset `{name}`')

    return sources

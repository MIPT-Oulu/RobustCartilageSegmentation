import logging

import torch
import numpy as np


logging.basicConfig()
logger = logging.getLogger('metrics')
logger.setLevel(logging.DEBUG)


def confusion_matrix(input_, target, num_classes):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py

    Args:
        input_: (d0, ..., dn) ndarray or tensor
        target: (d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.

    Returns:
        out: (num_classes, num_classes) ndarray
            Confusion matrix.
    """
    if torch.is_tensor(input_):
        input_ = input_.detach().to('cpu').numpy()
    if torch.is_tensor(target):
        target = target.detach().to('cpu').numpy()

    replace_indices = np.vstack((
        target.flatten(),
        input_.flatten())
    ).T
    cm, _ = np.histogramdd(
        replace_indices,
        bins=(num_classes, num_classes),
        range=[(0, num_classes-1), (0, num_classes-1)]
    )
    return cm.astype(np.uint32)


def dice_score_from_cm(cm):
    """
    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py

    Args:
        cm: (d, d) ndarray
            Confusion matrix.
    
    Returns:
        out: (d, ) list
            List of class Dice scores.
    """
    scores = []
    for index in range(cm.shape[0]):
        true_positives = cm[index, index]
        false_positives = cm[:, index].sum() - true_positives
        false_negatives = cm[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            score = 0
        else:
            score = 2 * float(true_positives) / denom
        scores.append(score)
    return scores


# ----------------------------------------------------------------------------


def _template_score(func_score_from_cm, input_, target, num_classes,
                    batch_avg, batch_weight, class_avg, class_weight):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    if torch.is_tensor(input_):
        num_samples = tuple(input_.size())[0]
    else:
        num_samples = input_.shape[0]

    scores = np.zeros((num_samples, num_classes))
    for sample_idx in range(num_samples):
        cm = confusion_matrix(input_=input_[sample_idx],
                              target=target[sample_idx],
                              num_classes=num_classes)
        scores[sample_idx, :] = func_score_from_cm(cm)

    if batch_avg:
        scores = np.mean(scores, axis=0, keepdims=True)
    if class_avg:
        if class_weight is not None:
            scores = scores * np.reshape(class_weight, (1, -1))
        scores = np.mean(scores, axis=1, keepdims=True)
    return np.squeeze(scores)


def dice_score(input_, target, num_classes,
               batch_avg=True, batch_weight=None,
               class_avg=False, class_weight=None):
    """

    Args:
        input_: (b, d0, ..., dn) ndarray or tensor
        target: (b, d0, ..., dn) ndarray or tensor
        num_classes: int
            Total number of classes.
        batch_avg: bool
            Whether to average over the batch dimension.
        batch_weight: (b,) iterable
            Batch samples importance coefficients.
        class_avg: bool
            Whether to average over the class dimension.
        class_weight: (c,) iterable
            Classes importance coefficients. Ignored when `class_avg` is False.

    Returns:
        out: scalar if `class_avg` is True, (num_classes,) list otherwise
    """
    return _template_score(
        dice_score_from_cm, input_, target, num_classes,
        batch_avg, batch_weight, class_avg, class_weight)

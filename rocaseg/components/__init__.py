from torch import nn
from torch import optim
from rocaseg.components.losses import CrossEntropyLoss
from rocaseg.components.metrics import (confusion_matrix, dice_score,
                                        dice_score_from_cm)
from rocaseg.components.checkpoint import CheckpointHandler


dict_losses = {
    'bce_loss': nn.BCEWithLogitsLoss,
    'multi_ce_loss': CrossEntropyLoss,
}


dict_metrics = {
    'confusion_matrix': confusion_matrix,
    'dice_score': dice_score,
    'bce_loss': nn.BCELoss(),
}


dict_optimizers = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
}


__all__ = [
    'dict_losses',
    'dict_metrics',
    'dict_optimizers',
    'confusion_matrix',
    'dice_score',
    'dice_score_from_cm',
    'CheckpointHandler',
]

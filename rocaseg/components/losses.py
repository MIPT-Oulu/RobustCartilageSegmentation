import logging

from torch import nn


logging.basicConfig()
logger = logging.getLogger('losses')
logger.setLevel(logging.DEBUG)


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, batch_avg=True, batch_weight=None,
                 class_avg=True, class_weight=None, **kwargs):
        """

        Parameters
        ----------
        batch_avg:
            Whether to average over the batch dimension.
        batch_weight:
            Batch samples importance coefficients.
        class_avg:
            Whether to average over the class dimension.
        class_weight:
            Classes importance coefficients.
        """
        super().__init__()
        self.num_classes = num_classes
        self.batch_avg = batch_avg
        self.class_avg = class_avg
        self.batch_weight = batch_weight
        self.class_weight = class_weight
        logger.warning('Redundant loss function arguments:\n{}'
                       .format(repr(kwargs)))
        self.ce = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, input_, target, **kwargs):
        """

        Parameters
        ----------
        input_: (b, ch, d0, d1) tensor
        target: (b, d0, d1) tensor

        Returns
        -------
        out: float tensor
        """
        return self.ce(input_, target)

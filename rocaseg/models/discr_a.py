from collections import OrderedDict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logging.basicConfig()
logger = logging.getLogger('models')
logger.setLevel(logging.DEBUG)


class DiscriminatorA(nn.Module):
    def __init__(self, basic_width=64, input_channels=5, output_channels=1,
                 restore_weights=False, path_weights=None, **kwargs):
        super().__init__()
        logger.warning('Redundant model init arguments:\n{}'
                       .format(repr(kwargs)))

        # Preparing the modules dict
        modules = OrderedDict()

        modules['conv1'] = \
            nn.Sequential(*[
                nn.Conv2d(input_channels, basic_width,
                          kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        modules['conv2'] = \
            nn.Sequential(*[
                nn.Conv2d(basic_width, basic_width*2,
                          kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        modules['conv3'] = \
            nn.Sequential(*[
                nn.Conv2d(basic_width*2, basic_width*4,
                          kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        modules['conv4'] = \
            nn.Sequential(*[
                nn.Conv2d(basic_width*4, basic_width*8,
                          kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        modules['output'] = nn.Conv2d(basic_width*8, output_channels,
                                      kernel_size=4, stride=2, padding=1)

        self.__dict__['_modules'] = modules
        if restore_weights:
            self.load_state_dict(torch.load(path_weights))

    def forward(self, x):
        tmp = x

        for name in self.__dict__['_modules']:
            layer = self.__dict__['_modules'][name]
            tmp = layer(tmp)

        out = F.interpolate(tmp, size=x.size()[-2:],
                            mode='bilinear', align_corners=True)
        return out

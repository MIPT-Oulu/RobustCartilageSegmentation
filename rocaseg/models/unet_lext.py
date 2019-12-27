"""
Dynamically created UNet with variable  Width, Depth and activation

Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


logging.basicConfig()
logger = logging.getLogger('models')
logger.setLevel(logging.DEBUG)


def depthwise_separable_conv(input_channels, output_channels):
    """
    """
    depthwise = nn.Conv2d(input_channels, input_channels,
                          kernel_size=3, padding=1, groups=input_channels)
    pointwise = nn.Conv2d(input_channels, output_channels,
                          kernel_size=1)
    return nn.Sequential(depthwise, pointwise)


def block_conv_bn_act(input_channels, output_channels,
                      convolution, activation):
    """
    """
    if convolution == 'regular':
        layer_conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
    elif convolution == 'depthwise_separable':
        layer_conv = depthwise_separable_conv(input_channels, output_channels)
    else:
        raise ValueError(f'Wrong `convolution`: {convolution}')

    if activation == 'relu':
        layer_act = nn.ReLU(inplace=True)
    elif activation == 'selu':
        layer_act = nn.SELU(inplace=True)
    elif activation == 'elu':
        layer_act = nn.ELU(1, inplace=True)
    else:
        raise ValueError(f'Wrong `activation`: {activation}')

    block = list()
    block.append(layer_conv)
    if activation == 'relu':
        block.append(nn.BatchNorm2d(output_channels))
    block.append(layer_act)

    return nn.Sequential(*block)


class Encoder(nn.Module):
    """Encoder class. for encoder-decoder architecture.

    """
    def __init__(self, input_channels, output_channels,
                 depth=2, convolution='regular', activation='relu'):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(block_conv_bn_act(input_channels, output_channels,
                                             convolution=convolution,
                                             activation=activation))
            else:
                tmp.append(block_conv_bn_act(output_channels, output_channels,
                                             convolution=convolution,
                                             activation=activation))

            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x):
        processed = self.layers(x)
        pooled = F.max_pool2d(processed, 2, 2)
        return processed, pooled


class Decoder(nn.Module):
    """Decoder class. for encoder-decoder architecture.

    """
    def __init__(self, input_channels, output_channels, depth=2, mode='bilinear',
                 convolution='regular', activation='relu'):
        super().__init__()
        self.layers = nn.Sequential()
        self.ups_mode = mode
        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(block_conv_bn_act(input_channels, output_channels,
                                             convolution=convolution,
                                             activation=activation))
            else:
                tmp.append(block_conv_bn_act(output_channels, output_channels,
                                             convolution=convolution,
                                             activation=activation))

            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x_big, x):
        x_ups = F.interpolate(x, size=x_big.size()[-2:], mode=self.ups_mode,
                              align_corners=True)
        y_cat = torch.cat([x_ups, x_big], 1)
        y = self.layers(y_cat)
        return y


class UNetLext(nn.Module):
    """UNet architecture with 3x3 convolutions. Created dynamically based on depth and width.

    """
    def __init__(self, basic_width=24, depth=6, center_depth=2,
                 input_channels=3, output_channels=1,
                 convolution='regular', activation='relu',
                 pretrained=False, path_pretrained=None,
                 restore_weights=False, path_weights=None, **kwargs):
        """

        Parameters
        ----------
        basic_width:
            Basic width of the network, which is doubled at each layer.
        depth:
            Number of layers.
        center_depth:
            Depth of the central block in UNet.
        input_channels:
            Number of input channels.
        output_channels:
            Number of output channels (/classes).
        convolution: {'regular', 'depthwise_separable'}
        activation: {'ReLU', 'SeLU', 'ELU'}
            Activation function.
        restore_weights: bool
            ???
        path_weights: str
            ???
        kwargs:
            Catches redundant arguments and issues a warning.
        """
        assert depth >= 2
        super().__init__()
        logger.warning('Redundant model init arguments:\n{}'
                       .format(repr(kwargs)))

        # Preparing the modules dict
        modules = OrderedDict()

        modules['down1'] = Encoder(input_channels, basic_width, activation=activation)

        # Automatically creating the Encoder based on the depth and width
        for level in range(2, depth + 1):
            mul_in = 2 ** (level - 2)
            mul_out = 2 ** (level - 1)
            layer = Encoder(basic_width * mul_in, basic_width * mul_out,
                            convolution=convolution, activation=activation)
            modules['down' + str(level)] = layer

        # Creating the center
        modules['center'] = nn.Sequential(
            *[block_conv_bn_act(basic_width * mul_out, basic_width * mul_out,
                                convolution=convolution, activation=activation)
              for _ in range(center_depth)]
            )

        # Automatically creating the decoder
        for level in reversed(range(2, depth + 1)):
            mul_in = 2 ** (level - 1)
            layer = Decoder(2 * basic_width * mul_in, basic_width * mul_in // 2,
                            convolution=convolution, activation=activation)
            modules['up' + str(level)] = layer

        modules['up1'] = Decoder(basic_width * 2, basic_width * 2,
                                 convolution=convolution, activation=activation)

        modules['mixer'] = nn.Conv2d(basic_width * 2, output_channels,
                                     kernel_size=1, padding=0, stride=1,
                                     bias=True)

        self.__dict__['_modules'] = modules
        if pretrained:
            self.load_state_dict(torch.load(path_pretrained))
        if restore_weights:
            self.load_state_dict(torch.load(path_weights))

    def forward(self, x):
        encoded_results = {}

        out = x
        for name in self.__dict__['_modules']:
            if name.startswith('down'):
                layer = self.__dict__['_modules'][name]
                convolved, pooled = layer(out)
                encoded_results[name] = convolved
                out = pooled

        out = self.center(out)

        for name in self.__dict__['_modules']:
            if name.startswith('up'):
                layer = self.__dict__['_modules'][name]
                out = layer(encoded_results['down' + name[-1]], out)
        return self.mixer(out)

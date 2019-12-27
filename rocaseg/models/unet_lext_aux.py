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


def ConvBlock3(inp, out, activation):
    """3x3 ConvNet building block with different activations support.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """
    if activation == 'relu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
    elif activation == 'selu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.SELU(inplace=True)
        )
    elif activation == 'elu':
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, padding=1),
            nn.ELU(1, inplace=True)
        )


class Encoder(nn.Module):
    """Encoder class. for encoder-decoder architecture.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """
    def __init__(self, input_channels, output_channels, depth=2, activation='relu'):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(ConvBlock3(input_channels, output_channels, activation))
            else:
                tmp.append(ConvBlock3(output_channels, output_channels, activation))

            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x):
        processed = self.layers(x)
        pooled = F.max_pool2d(processed, 2, 2)
        return processed, pooled


class Decoder(nn.Module):
    """Decoder class. for encoder-decoder architecture.

    Aleksei Tiulpin, Unversity of Oulu, 2017 (c).
    """
    def __init__(self, input_channels, output_channels, depth=2, mode='bilinear',
                 activation='relu'):
        super().__init__()
        self.layers = nn.Sequential()
        self.ups_mode = mode
        for i in range(depth):
            tmp = []
            if i == 0:
                tmp.append(ConvBlock3(input_channels, output_channels, activation))
            else:
                tmp.append(ConvBlock3(output_channels, output_channels, activation))

            self.layers.add_module('conv_3x3_{}'.format(i), nn.Sequential(*tmp))

    def forward(self, x_big, x):
        x_ups = F.interpolate(x, size=x_big.size()[-2:], mode=self.ups_mode,
                              align_corners=True)
        y_cat = torch.cat([x_ups, x_big], 1)
        y = self.layers(y_cat)
        return y


class AuxModule(nn.Module):
    def __init__(self, input_channels, output_channels, dilation_series, padding_series):
        super(AuxModule, self).__init__()
        self.layers = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.layers.append(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.layers:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.layers[0](x)
        for i in range(len(self.layers) - 1):
            out += self.layers[i + 1](x)
            return out


class UNetLextAux(nn.Module):
    """UNet architecture with 3x3 convolutions. Created dynamically based on depth and width.

    Aleksei Tiulpin, 2017 (c)
    """
    def __init__(self, basic_width=24, depth=6, center_depth=2,
                 input_channels=3, output_channels=1, activation='relu',
                 pretrained=False, path_pretrained=None,
                 restore_weights=False, path_weights=None,
                 with_aux=True, **kwargs):
        """

        Aleksei Tiulpin, Unversity of Oulu, 2017 (c).

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
        activation: {'ReLU', 'SeLU', 'ELU'}
            Activation function.
        restore_weights: bool
            ???
        path_weights: str
            ???
        kwargs:
        """
        assert depth >= 2
        super().__init__()
        logger.warning('Redundant model init arguments:\n{}'
                       .format(repr(kwargs)))
        self._with_aux = with_aux

        # Preparing the modules dict
        modules = OrderedDict()

        modules['down1'] = Encoder(input_channels, basic_width, activation=activation)

        # Automatically creating the Encoder based on the depth and width
        for level in range(2, depth + 1):
            mul_in = 2 ** (level - 2)
            mul_out = 2 ** (level - 1)
            layer = Encoder(basic_width * mul_in, basic_width * mul_out,
                            activation=activation)
            modules['down' + str(level)] = layer

        # Creating the center
        modules['center'] = nn.Sequential(
            *[ConvBlock3(basic_width * mul_out, basic_width * mul_out,
                         activation=activation)
              for i in range(center_depth)]
            )

        # Automatically creating the decoder
        for level in reversed(range(2, depth + 1)):
            mul_in = 2 ** (level - 1)
            layer = Decoder(2 * basic_width * mul_in, basic_width * mul_in // 2,
                            activation=activation)
            modules['up' + str(level)] = layer

            if self._with_aux and (level == 2):
                modules['aux'] = AuxModule(
                    input_channels=basic_width * mul_in // 2,
                    output_channels=output_channels,
                    dilation_series=[6, 12, 18, 24],
                    padding_series=[6, 12, 18, 24])

        modules['up1'] = Decoder(basic_width * 2, basic_width * 2,
                                 activation=activation)

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

                if name == 'up2':
                    out_aux = out

        if self._with_aux:
            out_aux = self.aux(out_aux)
            out_aux = F.interpolate(out_aux, size=x.size()[-2:],
                                    mode='bilinear', align_corners=True)
            out_main = self.mixer(out)
            return out_main, out_aux
        else:
            return self.mixer(out)

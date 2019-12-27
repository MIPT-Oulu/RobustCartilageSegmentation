import random
import cv2
import numpy as np
import math
import torch


class DualCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask=None):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class OneOf(object):
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        if self.state['p'] < self.prob:
            img, mask = self.state['t'](img, mask)
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()
        self.state['t'] = random.choice(self.transforms)
        self.state['t'].prob = 1.


class OneOrOther(object):
    def __init__(self, first, second, prob=.5):
        self.first = first
        first.prob = 1.
        self.second = second
        second.prob = 1.
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, x, mask=None):
        if self.state['p'] < self.prob:
            x, mask = self.first(x, mask)
        else:
            x, mask = self.second(x, mask)
        return x, mask

    def randomize(self):
        self.state['p'] = random.random()


class ImageOnly(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask=None):
        return self.transform(img, None)[0], mask


class NoTransform(object):
    def __call__(self, *args):
        return args


class ToTensor(object):
    def __call__(self, *args):
        return [torch.from_numpy(e) for e in args]


class VerticalFlip(object):
    def __init__(self, prob=.5):
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            img = np.flip(img, axis=1)
            if mask is not None:
                mask = np.flip(mask, axis=1)
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()


class HorizontalFlip(object):
    def __init__(self, prob=.5):
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            img = np.flip(img, axis=2)
            if mask is not None:
                mask = np.flip(mask, axis=2)
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()


class Flip(object):
    def __init__(self, prob=.5):
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            if self.state['d'] in (-1, 0):
                img = np.flip(img, axis=1)
            if self.state['d'] in (-1, 1):
                img = np.flip(img, axis=2)
            if mask is not None:
                if self.state['d'] in (-1, 0):
                    mask = np.flip(mask, axis=1)
                if self.state['d'] in (-1, 1):
                    mask = np.flip(mask, axis=2)
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()
        self.state['d'] = random.randint(-1, 1)


class Scale(object):
    def __init__(self, ratio_range=(0.7, 1.2), prob=.5):
        self.ratio_range = ratio_range
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            ch, d0_i, d1_i = img.shape
            d0_o = math.floor(d1_i * self.state['r'])
            d0_o = d0_o + d0_o % 2
            d1_o = math.floor(d1_i * self.state['r'])
            d1_o = d1_o + d1_o % 2

            # img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit,
            #                           borderType=cv2.BORDER_REFLECT_101)
            img = np.squeeze(img)
            img = cv2.resize(img, (d1_o, d0_o), interpolation=cv2.INTER_LINEAR)
            img = img[None, ...]

            if mask is not None:
                # msk1 = cv2.copyMakeBorder(mask, limit, limit, limit, limit,
                #                           borderType=cv2.BORDER_REFLECT_101)
                tmp = np.empty((mask.shape[0], d1_o, d0_o), dtype=mask.dtype)
                for idx_ch, mask_ch in enumerate(mask):
                    tmp[idx_ch] = cv2.resize(mask_ch, (d1_o, d0_o),
                                             interpolation=cv2.INTER_NEAREST)
                mask = tmp
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()
        self.state['r'] = round(random.uniform(*self.ratio_range), 2)


class Crop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        elif isinstance(output_size, tuple):
            self.output_size = output_size
        else:
            raise ValueError('Incorrect value')
        # self.keep_size = keep_size
        # self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        rows_in, cols_in = img.shape[1:]
        rows_out, cols_out = self.output_size
        rows_out = min(rows_in, rows_out)
        cols_out = min(cols_in, cols_out)

        r0 = math.floor(self.state['r0f'] * (rows_in - rows_out))
        c0 = math.floor(self.state['c0f'] * (cols_in - cols_out))
        r1 = r0 + rows_out
        c1 = c0 + cols_out

        img = np.ascontiguousarray(img[:, r0:r1, c0:c1])
        if mask is not None:
            mask = np.ascontiguousarray(mask[:, r0:r1, c0:c1])
        return img, mask

    def randomize(self):
        # self.state['p'] = random.random()
        self.state['r0f'] = random.random()
        self.state['c0f'] = random.random()


class CenterCrop(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        c, h, w = img.shape
        dy = (h - self.height) // 2
        dx = (w - self.width) // 2

        y1 = dy
        y2 = y1 + self.height
        x1 = dx
        x2 = x1 + self.width
        img = np.ascontiguousarray(img[:, y1:y2, x1:x2])
        if mask is not None:
            mask = np.ascontiguousarray(mask[:, y1:y2, x1:x2])

        return img, mask


class Pad(object):
    def __init__(self, dr, dc, **kwargs):
        self.dr = dr
        self.dc = dc
        self.kwargs = kwargs

    def __call__(self, img, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        pad_width = (0, 0), (self.dr,) * 2, (self.dc,) * 2
        img = np.pad(img, pad_width, **self.kwargs)
        if mask is not None:
            mask = np.pad(mask, pad_width, **self.kwargs)
        return img, mask


class GammaCorrection(object):
    def __init__(self, gamma_range=(0.5, 2), prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, image, mask=None):
        """

        Parameters
        ----------
        img: (ch, d0, d1) ndarray
        mask: (ch, d0, d1) ndarray
        """
        if self.state['p'] < self.prob:
            image = image ** (1 / self.state['gamma'])
            # TODO: implement also for integers
            image = np.clip(image, 0, 1)
        return image, mask

    def randomize(self):
        self.state['p'] = random.random()
        self.state['gamma'] = random.uniform(*self.gamma_range)


class BilateralFilter(object):
    def __init__(self, d, sigma_color, sigma_space, prob=.5):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.prob = prob

        self.state = dict()
        self.randomize()

    def __call__(self, img, mask=None):
        if self.state['p'] < self.prob:
            img = np.squeeze(img)
            img = cv2.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space)
            img = img[None, ...]
        return img, mask

    def randomize(self):
        self.state['p'] = random.random()

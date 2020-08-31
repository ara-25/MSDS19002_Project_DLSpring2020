import operator
import pathlib
import re
import cv2
import numpy as np
from functools import reduce
from PIL import Image, ImageOps


import torchvision.transforms.functional as FF
import numpy as np


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def get_latest_file_iteration(folder, pattern='*'):
    folder = pathlib.Path(folder)
    matches = [(fp, re.findall(r'\d+', fp.stem)) for fp in folder.glob(pattern)]
    file_itr_pairs = [(fp, int(m[-1])) for fp, m in matches if len(m) > 0]
    if len(file_itr_pairs) == 0:
        return None, None
    return max(file_itr_pairs, key=lambda t: t[1])


def dict_from_module(module):
    return {k: getattr(module, k) for k in module.__all__}


class Invert(object):
    """Inverts the color channels of an PIL Image
    while leaving intact the alpha channel.
    """
    
    def invert(self, img):
        r"""Invert the input PIL Image.
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        if not FF._is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        cv_image = np.asarray(img)
        ret, thresh = cv2.threshold(cv_image,0,255,cv2.THRESH_BINARY_INV+
                                    cv2.THRESH_OTSU)
        return Image.fromarray(thresh)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be inverted.
        Returns:
            PIL Image: Inverted image.
        """
        return self.invert(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

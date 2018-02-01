from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import torchvision.transforms as transforms

class randomcrop_yh(transforms.RandomCrop):
    """Crop the given PIL.Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, images):
        """
        Args:
            img (PIL.Image): Image to be cropped.

        Returns:
            PIL.Image: Cropped image.
        """
        output = []
        for i in range(len(images)):
            img = images[i]
            if self.padding > 0:
                img = ImageOps.expand(img, border=self.padding, fill=0)

            w, h = img.size
            th, tw = self.size
            if w == tw and h == th:
                return img
            if i==0:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
            output.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return output
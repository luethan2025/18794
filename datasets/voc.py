import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from .utils import voc_cmap

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``.
    """
    # color map for each category
    cmap = voc_cmap()

    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        
        self.image_set = image_set
        base_dir = "VOCdevkit/VOC2012"
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Please check your data download.')
        
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image for visualization, using the color map"""
        return cls.cmap[mask]

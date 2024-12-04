import torch.utils.data as data
import os
import numpy as np
import matplotlib
from PIL import Image

def coco_cmap(stuff_start_id=92, stuff_end_id=182, cmap_name='jet', add_things=True, add_unlabeled=True, add_other=True):
    label_count = stuff_end_id - stuff_start_id + 1
    cmap_gen = matplotlib.cm.get_cmap(cmap_name, label_count)
    cmap = cmap_gen(np.arange(label_count))
    cmap = cmap[:, 0:3]

    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(label_count)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    if add_things:
        thingsPadding = np.zeros((stuff_start_id - 1, 3))
        cmap = np.vstack((thingsPadding, cmap))

    if add_unlabeled:
        cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

    if add_other:
        cmap = np.vstack((cmap, (1.0, 0.843, 0.0)))

    return cmap

class COCOSegmentation(data.Dataset):
    # color map for each category
    cmap = coco_cmap()
    
    def __init__(self,
                 root,
                 image_set='train',
                 transform=None):

        self.root = os.path.expanduser(root)
        self.transform = transform
        
        self.image_set = image_set
        base_dir = "COCOdevkit/COCO"
        coco_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(coco_root, 'JPEGImages')

        if not os.path.isdir(coco_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Please check your data download.')
        
        mask_dir = os.path.join(coco_root, 'SegmentationClass')
        splits_dir = os.path.join(coco_root, 'ImageSets/Segmentation')
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

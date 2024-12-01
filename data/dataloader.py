from torch.utils.data import Dataset
import numpy as np
import os
import os.path
from PIL import Image

class SegmentationDataset(Dataset):
    def __init__(self, base_dir, images_dir="Images", mask_dir="Mask", transforms=None):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

        self.images = []
        self.masks = []

        for class_name in os.listdir(base_dir):
            class_images_dir = os.path.join(base_dir, class_name, images_dir)
            self.images.extend([
                os.path.join(class_images_dir, file)
                    for file in os.listdir(class_images_dir) if file.endswith('jpg')
            ])

            class_mask_dir = os.path.join(base_dir, class_name, mask_dir)
            self.masks.extend([
                os.path.join(class_mask_dir, file)
                    for file in os.listdir(class_mask_dir) if file.endswith('png')
            ])
    
        assert (len(self.images) == len(self.masks))

    def __len__(self):
        return len(self.masks)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        target = Image.open(self.masks[idx])
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

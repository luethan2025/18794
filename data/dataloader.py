import os
import os.path

import cv2
import torch
import torch.utils.data as data

class CMU_GO(data.Dataset):
    """Dataset iterator.

    Parameters
    ----------
    dataset_path: string
        Path to parent directory with subdirectories containing images.
    use_rescaled_images: bool
        Load the rescaled images (defaults to False).
    """
    def __init__(self, dataset_path, use_rescaled_images=False):
        super(CMU_GO, self).__init__()
        if use_rescaled_images:
            dataset_root_path = os.path.join(dataset_path, "rescaled")
        else:
            dataset_root_path = os.path.join(dataset_path, "dataset")

        self.dirs_path = [
            os.path.join(dataset_root_path, dir)
                for dir in os.listdir(dataset_root_path)
        ]

        self.labels = [
            os.path.split(dir_path)[-1]
                for dir_path in self.dirs_path
        ]

        self.imgs_path = [
        os.path.join(dir_path, img_path)
            for dir_path in self.dirs_path
                for img_path in os.listdir(dir_path) if img_path.endswith((".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        img = cv2.imread(img_path)

        head, _ = os.path.split(img_path)
        _, label = os.path.split(head)
        if label not in self.labels:
            raise ValueError(f"saw {label} expected a label from {self.labels}")

        return torch.from_numpy(img), label

    def __repr__(self):
        header = "labels:"
        labels = "\n".join(
                    " ".join(f"{label:<15}"
                        for label in self.labels[i:i + 5])
                            for i in range(0, len(self.labels), 5)
                    )
        return header + labels

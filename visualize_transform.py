import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from datasets import Toy
from utils import ext_transforms as et

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--base_dir", type=str, default='VOCdevkit/VOC2012',
                        help="path to Dataset root")
    parser.add_argument("--data_type", type=str, default='train',
                        choices=['train', 'val'], help="train or validation")
    parser.add_argument("--num_img", type=int, default=1,
                        help="number of images to sample")
    parser.add_argument("--crop_size", type=int, default=224)
    
    return parser


def main():
    opts = get_argparser().parse_args()
    transform = et.ExtCompose([
        et.ExtRandomScale((0.5, 2)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomBox(0, (50, 100), p=0.85),
        et.ExtRandomBox(0, (50, 100), p=0.85),
        et.ExtToTensor()
    ])
    dst = Toy(root=opts.data_root, base_dir=opts.base_dir, image_set=opts.data_type, num_img=opts.num_img, transform=transform)

    def plot_images(images, rows, cols, figsize=(20,7)):
        assert len(images)==rows*cols

        fig, axarr = plt.subplots(rows, cols, figsize=figsize)
        axarr = np.array(axarr).reshape(-1) 

        for i, ax in enumerate(axarr):
            ax.imshow(images[i])
            ax.axis('off')

        plt.tight_layout()
        plt.show()
    
    loader = data.DataLoader(dst, batch_size=1, shuffle=False)

    for (image, label) in loader:
        img, lbl = np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0)), \
                    np.squeeze(label.cpu().numpy())
        plot_images([img, lbl], rows=1, cols=2)

if __name__ == '__main__':
    main()

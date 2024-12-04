import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from datasets import VOCToy
from utils import ext_transforms as et

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--data_type", type=str, default='train',
                        choices=['train', 'val'], help="train or validation")
    parser.add_argument("--num_img", type=int, default=1,
                        help="number of images to sample")
    
    return parser


def main():
    opts = get_argparser().parse_args()
    transform = et.ExtCompose([
        et.ExtRandomBox(0, (200, 250)),
        et.ExtToTensor()
    ])
    dst = VOCToy(root=opts.data_root, image_set=opts.data_type, num_img=opts.num_img, transform=transform)

    def plot_images(images, rows, cols, figsize=(20,7)):
        assert len(images)==rows*cols

        fig, axarr = plt.subplots(rows, cols, figsize=figsize)
        axarr = np.array(axarr).reshape(-1) 

        for i, ax in enumerate(axarr):
            ax.imshow(images[i], cmap='gray')
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

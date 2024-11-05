from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os.path
import cv2

from dataloader import CMU_GO

idx = 0
imgs = []
normalized_imgs = []
axarr = None

def on_key(event):
    global idx, imgs, axarr
    if event.key == 'right':
        idx = (idx + 1) % len(imgs)
        plot_images([imgs[idx], normalized_imgs[idx]], 1, 2)
    elif event.key == 'left':
        idx = (idx - 1) % len(imgs)
        plot_images([imgs[idx], normalized_imgs[idx]], 1, 2)

def plot_images(images, rows, cols, figsize=(8, 4)):
    assert (len(images)==rows*cols)
    global axarr
    if axarr is None:
        fig, axarr = plt.subplots(rows, cols, figsize=figsize)
        axarr = np.array(axarr).reshape(-1) 
        fig.canvas.mpl_connect('key_press_event', on_key)

    for i, ax in enumerate(axarr):
        ax.imshow(images[i], extent=[0, 800, 0, 800])
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 800)
        ax.axis('off')
    plt.draw()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='visualizer')
    parser.add_argument('-f', '--filename', required=True)
    args = parser.parse_args()
    filename = args.filename
    assert (os.path.exists(filename))

    statistics = np.load(filename)
    mean = np.array([
        statistics['avg_red'].item(),
        statistics['avg_blue'].item(),
        statistics['avg_green'].item()
    ])
    std = np.array([
        statistics['std_red'].item(),
        statistics['std_blue'].item(),
        statistics['std_green'].item(),
    ])

    dataset = DataLoader(CMU_GO(), batch_size=1, shuffle=True)
    for img, label in dataset:
        img = img.squeeze(0).numpy()
        imgs.append(img)

        normalized_img = ((img / 255.0) - mean[None, None, :]) / std[None, None, :]
        normalized_imgs.append(np.clip(normalized_img, 0, 1))
    plot_images([imgs[idx], normalized_imgs[idx]], 1, 2)

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def plot_best_worst(exp_img, imgs, exp, unc, dpi, num_images=25, save=None):
    image_dir = exp_img + imgs
    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))
    order = np.argsort(np.load(exp + unc))
    ordered = [image_files[i] for i in order]

    grid = int(np.sqrt(num_images))
    best = ordered[:num_images]
    worst = ordered[::-1][:num_images]

    fig = plt.figure(figsize=(grid * 5, grid * 2.6))
    blocks = fig.subfigures(1, 2, wspace=0.04)
    for sf, title, files in [(blocks[0], "Low uncertainty", best),
                             (blocks[1], "High uncertainty", worst)]:
        axes = sf.subplots(grid, grid)
        for ax, f in zip(axes.ravel(), files):
            ax.imshow(Image.open(os.path.join(image_dir, f)), aspect="auto")
            ax.axis("off")
        sf.subplots_adjust(wspace=0, hspace=0, top=0.93, bottom=0, left=0, right=1)
        sf.suptitle(title, fontsize=24, y=0.965)
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

FS_CLASS = 16
FS_KIND = 13
FS_COL = 11


def plot_perturbations(exps, unc, M=5, names=None, dpi=200, save="qual_ensemble.png"):
    if isinstance(exps, str):
        exps = [exps]
    if names is None:
        names = [f"class {k}" for k in range(len(exps))]

    rows = []
    for exp in exps:
        o = np.argsort(np.load(f"{exp}{unc}"))
        rows.append(("low", exp, int(o[0])))
        rows.append(("high", exp, int(o[-1])))

    nrows, ncols = len(rows), M + 1
    L, R, T, B = 0.13, 0.99, 0.93, 0.01
    figW = ncols * 1.6
    cell = figW * (R - L) / ncols
    figH = cell * nrows / (T - B)

    fig, axes = plt.subplots(nrows, ncols, figsize=(figW, figH))
    for r, (kind, exp, i) in enumerate(rows):
        for m in range(ncols):
            ax = axes[r][m]
            ax.imshow(Image.open(f"{exp}/{m}/imgs/{i:05d}.png"), aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title("base" if m == 0 else str(m), fontsize=FS_COL)
        axes[r][0].set_ylabel(kind, fontsize=FS_KIND, rotation=90, va="center", labelpad=6)

    fig.subplots_adjust(left=L, right=R, top=T, bottom=B, wspace=0, hspace=0)

    for c, name in enumerate(names):
        p_top = axes[2 * c][0].get_position()
        p_bot = axes[2 * c + 1][0].get_position()
        y_mid = (p_top.y1 + p_bot.y0) / 2
        fig.text(0.045, y_mid, name, rotation=90, va="center", ha="center", fontsize=FS_CLASS)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        print(f"saved {save}")
    plt.show()

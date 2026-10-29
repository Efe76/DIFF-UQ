import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
})

methods = ["Gen.\nuncertainty", "Labelled\nCLIP", "Label-free\nCLIP"]
colors  = ["#1f77b4", "#2ca02c", "#9467bd"]

fid = [7.99, 6.95, 7.26]
pre = [0.674, 0.687, 0.690]
rec = [0.713, 0.697, 0.690]

fig, axes = plt.subplots(1, 3, figsize=(34, 11))
plt.subplots_adjust(bottom=0.14, wspace=0.20)

x = list(range(len(methods)))

panels = [
    (fid, r"FID ($\downarrow$)",     (6.0, 8.5),  0.5,  "{:.2f}"),
    (pre, r"Precision ($\uparrow$)", (0.66, 0.70), 0.01, "{:.3f}"),
    (rec, r"Recall ($\uparrow$)",    (0.68, 0.72), 0.01, "{:.3f}"),
]

for ax, (vals, title, ylim, loc, fmt) in zip(axes, panels):
    ax.bar(x, vals, color=colors, width=0.6, zorder=3,
           edgecolor="black", linewidth=1.5)

    for xi, v in zip(x, vals):
        ax.annotate(fmt.format(v), (xi, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=20)

    ax.set_title(title, fontsize=28, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=20)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(loc))
    ax.grid(True, axis="y", linestyle="--", color="#e0e0e0", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=22, width=2.0, length=7)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

fig.suptitle("Zero-order search over initial noise: generative uncertainty vs. CLIP verifiers "
             "(10K samples)", fontsize=30, y=1.02)

plt.savefig("rq2_search.png", dpi=300, bbox_inches="tight")
print("Plot saved as: rq2_search.png")

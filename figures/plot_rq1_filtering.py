import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family":      "serif",
    "mathtext.fontset": "cm",
})

N_vals   = [6, 7, 8, 9, 10, 11, 12]
x_labels = ["6K", "7K", "8K", "9K", "10K", "11K", "12K"]

rand_fid = [13.0578, 12.1875, 11.5566, 11.0551, 10.6587, 10.3003, 9.97205]
rand_pre = [0.61967, 0.61886, 0.61888, 0.62111, 0.62140, 0.62364, 0.62242]
rand_rec = [0.76112, 0.75388, 0.75078, 0.74438, 0.73810, 0.73258, 0.72694]

lp_fid = [12.2189, 10.4833, 9.50618, 8.94964, 8.76375, 9.05909, 9.89822]
lp_pre = [0.70050, 0.68743, 0.67363, 0.65978, 0.64900, 0.63700, 0.62258]
lp_rec = [0.70286, 0.70720, 0.71658, 0.71984, 0.72256, 0.72404, 0.72642]

p50_fid = [13.3172, 11.3498, 9.99671, 9.21395, 8.86148, 9.00712, 9.90486]
p50_pre = [0.70233, 0.69114, 0.67925, 0.66511, 0.65150, 0.63755, 0.62308]
p50_rec = [0.69170, 0.70344, 0.71008, 0.71458, 0.71872, 0.72300, 0.72624]

p25_fid = [13.3343, 11.3230, 10.0141, 9.24026, 8.93072, 9.05031, 9.90901]
p25_pre = [0.69817, 0.68800, 0.67913, 0.66544, 0.65050, 0.63764, 0.62300]
p25_rec = [0.69564, 0.70290, 0.70684, 0.71360, 0.71956, 0.72280, 0.72610]

p10_fid = [12.9818, 11.1388, 10.0009, 9.29072, 9.04536, 9.17001, 9.92905]
p10_pre = [0.69783, 0.68371, 0.67263, 0.65867, 0.64710, 0.63682, 0.62308]
p10_rec = [0.70162, 0.70772, 0.71138, 0.71738, 0.72176, 0.72400, 0.72544]

p5_fid = [13.7758, 12.6036, 11.6875, 11.1160, 10.6075, 10.2257, 9.94028]
p5_pre = [0.63833, 0.63557, 0.63225, 0.63156, 0.62940, 0.62473, 0.62292]
p5_rec = [0.73884, 0.74230, 0.74024, 0.73652, 0.73076, 0.72882, 0.72628]

fig, axes = plt.subplots(1, 3, figsize=(40, 11))
plt.subplots_adjust(bottom=0.19, wspace=0.16)

line_kwargs = {"linewidth": 5.5, "marker": "o", "markersize": 10}

styles = {
    "Random baseline":           ("#7f7f7f", "--"),
    "Laplace (T=50)":            ("#2ca02c", "-"),
    "Input perturbation (T=50)": ("#1f77b4", "-"),
    "Input perturbation (T=25)": ("#9467bd", "-"),
    "Input perturbation (T=10)": ("#ff7f0e", "-"),
    "Input perturbation (T=5)":  ("#d62728", "-"),
}

panels = [
    (rand_fid, lp_fid, p50_fid, p25_fid, p10_fid, p5_fid,
     r"FID ($\downarrow$)", (8.0, 14.0), 1.0),
    (rand_pre, lp_pre, p50_pre, p25_pre, p10_pre, p5_pre,
     r"Precision ($\uparrow$)", (0.60, 0.72), 0.02),
    (rand_rec, lp_rec, p50_rec, p25_rec, p10_rec, p5_rec,
     r"Recall ($\uparrow$)", (0.68, 0.77), 0.01),
]

for ax, (rnd, lp, p50, p25, p10, p5, title, ylim, loc) in zip(axes, panels):
    series = [
        ("Random baseline", rnd),
        ("Laplace (T=50)", lp),
        ("Input perturbation (T=50)", p50),
        ("Input perturbation (T=25)", p25),
        ("Input perturbation (T=10)", p10),
        ("Input perturbation (T=5)", p5),
    ]
    for name, y in series:
        color, ls = styles[name]
        ax.plot(N_vals, y, label=name, color=color, linestyle=ls, **line_kwargs)

    ax.set_title(title, fontsize=28, pad=12)
    ax.set_ylim(ylim)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(loc))

fig.suptitle("Generative-uncertainty filtering: input perturbation vs. Laplace",
             fontsize=34, y=1.02)

for ax in axes:
    ax.set_xticks(N_vals)
    ax.set_xticklabels(x_labels)
    ax.grid(True, linestyle="--", color="#e0e0e0", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=22, width=2.0, length=7)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

fig.text(0.5, 0.105, "Number of images retained after filtering (out of 12K)",
         ha="center", fontsize=28)

handles, labels = axes[0].get_legend_handles_labels()
hmap = dict(zip(labels, handles))
blank = Line2D([], [], linestyle="none", marker="none", label="")

ordered_handles = [
    blank,                            hmap["Input perturbation (T=50)"],
    hmap["Random baseline"],          hmap["Input perturbation (T=25)"],
    hmap["Laplace (T=50)"],           hmap["Input perturbation (T=10)"],
    blank,                            hmap["Input perturbation (T=5)"],
]
ordered_labels = [h.get_label() for h in ordered_handles]

leg = fig.legend(ordered_handles, ordered_labels, loc="lower center", ncol=4,
                 bbox_to_anchor=(0.5, -0.055), frameon=True, fontsize=30,
                 handlelength=4, markerscale=1.6)
leg.get_frame().set_linewidth(2.0)

plt.savefig("rq1_filtering.png", dpi=300, bbox_inches="tight")
print("Plot saved as: rq1_filtering.png")

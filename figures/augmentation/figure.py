import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

font = {"family": "CMU Serif", "size": 18}
plt.rc("font", **font)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

x = [1, 2, 3, 4]

pol_color = "#1b7837"
die_color = "#b35900"
mean_color = "#002db3"

datadir = "data"

with open(os.path.join(datadir, "BN_polarization_augmentation.pkl"), "rb") as f:
    data = pickle.load(f)

errors = data["maes"]
augmented_train_data = errors[np.array([1, 3, 5, 7])]
non_augmented_train_data = errors[np.array([0, 2, 4, 6])]

augmented_train = np.mean(augmented_train_data, axis=1)
non_augmented_train = np.mean(non_augmented_train_data, axis=1)
augmented_train_std = np.std(augmented_train_data, axis=1)
non_augmented_train_std = np.std(non_augmented_train_data, axis=1)
mean_performance = [0.0001733, 0.018314]

# mean_performance = [1e-2, 1e-2, 1e-1, 1e-1]

fig = plt.figure(figsize=(10, 8))

gs = fig.add_gridspec(nrows=1, ncols=1, left=0.08, right=0.93, top=0.98, bottom=0.11)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_yscale("log")
ax1.xaxis.set_ticks([1, 2, 3, 4])
ax1.xaxis.set_ticklabels(
    [
        "Original\nArchitecture",
        "Biased\nArchitecture",
        "Original\nArchitecture",
        "Biased\nArchitecture",
    ]
)
ax1.tick_params(labelsize=25)
ax1.grid(which="major", axis="y")

ax1.plot([2.5, 2.5], [2e-6, 0.05], ls="--", color="k")
ax1.set_ylim([1e-6, 0.9])

aug_train_plot = ax1.errorbar(
    x[:2],
    augmented_train[:2],
    yerr=augmented_train_std[:2],
    ls="--",
    marker="x",
    color=pol_color,
    markersize=6,
    linewidth=2,
)
aug_train_plot2 = ax1.errorbar(
    x[2:],
    augmented_train[2:],
    yerr=augmented_train_std[2:],
    ls="--",
    marker="x",
    markersize=6,
    color=pol_color,
    linewidth=2,
)
non_aug_train_plot = ax1.errorbar(
    x[:2],
    non_augmented_train[:2],
    yerr=non_augmented_train_std[:2],
    ls="-.",
    marker="d",
    markersize=6,
    color=die_color,
    linewidth=2,
)
non_aug_train_plot2 = ax1.errorbar(
    x[2:],
    non_augmented_train[2:],
    yerr=non_augmented_train_std[2:],
    ls="-.",
    marker="d",
    markersize=6,
    color=die_color,
    linewidth=2,
)
(mean_train_plot,) = ax1.plot(
    x[:2],
    [mean_performance[0], mean_performance[0]],
    ls="-",
    linewidth=3,
    color=mean_color,
)
(mean_train_plot2,) = ax1.plot(
    x[2:],
    [mean_performance[1], mean_performance[1]],
    ls="-",
    linewidth=3,
    color=mean_color,
)

ax1.legend(
    [mean_train_plot, aug_train_plot, non_aug_train_plot],
    ["Average of Testing Set", "Augmented Training Set", "Original Training Set"],
    loc="upper center",
)
fig.text(
    ax1.get_position().get_points()[0, 0] + 0.035,
    ax1.get_position().get_points()[1, 1] - 0.08,
    "Original Test Set",
    color="k",
    fontsize=20,
)

fig.text(
    ax1.get_position().get_points()[1, 0] - 0.24,
    ax1.get_position().get_points()[1, 1] - 0.08,
    "Augmented Test Set",
    color="k",
    fontsize=20,
)
ax1.arrow(1.5, 0.18, -0.7, 0, head_width=0.05, color="k")
ax1.arrow(3.3, 0.18, 0.7, 0, head_width=0.05, color="k")

# plt.show()
plt.savefig("augmented_results.eps")

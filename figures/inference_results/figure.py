import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from cmocean import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


data_dir = "data"
detailed_data_path = os.path.join(data_dir, "temp_detailed_inference.pkl")

with open(detailed_data_path, "rb") as f:
    detailed_data = pickle.load(f)

font = {"family": "CMU Serif", "size": 18}
plt.rc("font", **font)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")
# plt.rcParams["axes.formatter.limits"] = (0, 0)

BN_pol_trues = detailed_data["BN_polarization_trues"]
BN_pol_preds = detailed_data["BN_polarization_preds"]
BN_die_trues = detailed_data["BN_dielectric_trues"]
BN_die_preds = detailed_data["BN_dielectric_preds"]
GaAs_pol_trues = detailed_data["GaAs_polarization_trues"]
GaAs_pol_preds = detailed_data["GaAs_polarization_preds"]
GaAs_die_trues = detailed_data["GaAs_dielectric_trues"]
GaAs_die_preds = detailed_data["GaAs_dielectric_preds"]

BN_pol_true_norms = np.linalg.norm(BN_pol_trues, 2, axis=-1)
BN_pol_pred_norms = np.linalg.norm(BN_pol_preds, 2, axis=-1)
BN_pol_errors = BN_pol_preds - BN_pol_trues
BN_pol_error_norms = np.linalg.norm(BN_pol_errors, 2, axis=-1)
BN_pol_relative_errors = BN_pol_error_norms / BN_pol_true_norms

BN_die_true_norms = np.linalg.norm(BN_die_trues, 2, axis=-1)
BN_die_pred_norms = np.linalg.norm(BN_die_preds, 2, axis=-1)
BN_die_errors = BN_die_preds - BN_die_trues
BN_die_error_norms = np.linalg.norm(BN_die_errors, 2, axis=-1)
BN_die_relative_errors = BN_die_error_norms / BN_die_true_norms

GaAs_pol_true_norms = np.linalg.norm(GaAs_pol_trues, 2, axis=-1)
GaAs_pol_pred_norms = np.linalg.norm(GaAs_pol_preds, 2, axis=-1)
GaAs_pol_errors = GaAs_pol_preds - GaAs_pol_trues
GaAs_pol_error_norms = np.linalg.norm(GaAs_pol_errors, 2, axis=-1)
GaAs_pol_relative_errors = GaAs_pol_error_norms / GaAs_pol_true_norms

GaAs_die_true_norms = np.linalg.norm(GaAs_die_trues, 2, axis=-1)
GaAs_die_pred_norms = np.linalg.norm(GaAs_die_preds, 2, axis=-1)
GaAs_die_errors = GaAs_die_preds - GaAs_die_trues
GaAs_die_error_norms = np.linalg.norm(GaAs_die_errors, 2, axis=-1)
GaAs_die_relative_errors = GaAs_die_error_norms / GaAs_die_true_norms

pol_max = max(BN_pol_relative_errors.max(), GaAs_pol_relative_errors.max())
die_max = max(BN_die_relative_errors.max(), GaAs_die_relative_errors.max())


fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(
    nrows=5,
    ncols=8,
    width_ratios=[0.01, 1, 0.2, 0.7, 0.15, 0.7, 0.05, 0.05],
    height_ratios=[0.1, 1, 0.3, 1, 0.06],
    hspace=0,
    wspace=0,
    left=0.05,
    right=0.93,
    top=0.97,
    bottom=0.07,
)

fig.text(0.01, 0.93, "a)", fontsize=30)
ax1 = fig.add_subplot(gs[1:4, 1])
ax1.grid(linestyle="-.")
ax1.set_ylabel("Test MAE")
ax1.set_xlabel("Dataset size")


cbax1 = plt.subplot(gs[1, 7])
cbax2 = plt.subplot(gs[3, 7])

ax2 = fig.add_subplot(gs[1, 3])
ax2.grid(linestyle="-.")

im2 = ax2.scatter(
    BN_pol_true_norms,
    BN_pol_error_norms,
    c=cm.haline_r(BN_pol_relative_errors / pol_max),
)
im2.set_cmap(cm.haline_r)
im2.set_clim(0, pol_max)

cbar = plt.colorbar(im2, cax=cbax1, use_gridspec=True)
cbar.set_ticks([0, 0.0001, 0.0002])
cbar.set_ticklabels(["0.00\%", "0.01\%", "0.02\%"])
cbar.set_label("Relative error")

ax2.set_ylim([0, BN_pol_error_norms.max() * 3])
ax2.set_xlabel(r"$|\bm{P}_{true}| [me/a^{2}_0]$")
ax2.set_ylabel(r"$|\bm{P}_{true} - \bm{P}_{pred}| [me/a^{2}_0]$")
ax2.set_title("BN", fontsize=30)

ax22 = inset_axes(ax2, width="55%", height="50%")
ax22.scatter(
    BN_pol_true_norms,
    BN_pol_pred_norms,
    c=cm.haline_r(BN_pol_relative_errors / pol_max),
)
ax22.grid(linestyle="-.")
ax22.plot(
    [BN_pol_true_norms.min(), BN_pol_true_norms.max()],
    [BN_pol_true_norms.min(), BN_pol_true_norms.max()],
    "--",
    c="gray",
)
ax22.tick_params(axis="both", labelsize=10)

ax3 = fig.add_subplot(gs[1, 5])
ax3.grid(linestyle="-.")
ax3.set_xlabel(r"$|\bm{P}_{true}| [me/a^{2}_0]$")
ax3.set_title("GaAs", fontsize=30)

im3 = ax3.scatter(
    GaAs_pol_true_norms,
    GaAs_pol_error_norms,
    c=cm.haline_r(GaAs_pol_relative_errors / pol_max),
)
im3.set_cmap(cm.haline_r)
im3.set_clim(0, pol_max)
ax3.set_ylim([0, GaAs_pol_error_norms.max() * 3])

ax32 = inset_axes(ax3, width="55%", height="50%")
ax32.scatter(
    GaAs_pol_true_norms,
    GaAs_pol_pred_norms,
    c=cm.haline_r(GaAs_pol_relative_errors / pol_max),
)
ax32.grid(linestyle="-.")
ax32.plot(
    [GaAs_pol_true_norms.min(), GaAs_pol_true_norms.max()],
    [GaAs_pol_true_norms.min(), GaAs_pol_true_norms.max()],
    "--",
    c="gray",
)
ax32.tick_params(axis="both", labelsize=10)

ax4 = fig.add_subplot(gs[3, 3])
ax4.grid(linestyle="-.")
ax4.set_xlabel(r"$|\bm{\epsilon}^\infty_{true}| [\epsilon_0]$")
ax4.set_ylabel(
    r"$|\bm{\epsilon}^\infty_{true} - \bm{\epsilon}^\infty_{pred}| [\epsilon_0]$"
)
ax4.ticklabel_format(axis="y", scilimits=(0, 0))

im4 = ax4.scatter(
    BN_die_true_norms,
    BN_die_error_norms,
    c=cm.haline_r(BN_die_relative_errors / die_max),
)
im4.set_cmap(cm.haline_r)
im4.set_clim(0, die_max)

ax4.set_ylim([0, BN_die_error_norms.max() * 3])

ax42 = inset_axes(ax4, width="55%", height="50%")
ax42.scatter(
    BN_die_true_norms,
    BN_die_pred_norms,
    c=cm.haline_r(BN_die_relative_errors / die_max),
)
ax42.grid(linestyle="-.")
ax42.plot(
    [BN_die_true_norms.min(), BN_die_true_norms.max()],
    [BN_die_true_norms.min(), BN_die_true_norms.max()],
    "--",
    c="gray",
)
ax42.tick_params(axis="both", labelsize=10)

ax5 = fig.add_subplot(gs[3, 5])
ax5.grid(linestyle="-.")
ax5.set_xlabel(r"$|\bm{\epsilon}^\infty_{true}| [\epsilon_0]$")

im5 = ax5.scatter(
    GaAs_die_true_norms,
    GaAs_die_error_norms,
    c=cm.haline_r(GaAs_die_relative_errors / die_max),
)
im5.set_cmap(cm.haline_r)
im5.set_clim(0, die_max)

cbar = plt.colorbar(im5, cax=cbax2, use_gridspec=True, format="%.1e")
cbar.set_label("Relative error")
cbar.set_ticks([0, 0.001, 0.002, 0.003])
cbar.set_ticklabels(["0.0\%", "0.1\%", "0.2\%", "0.3\%"])
ax5.set_ylim([0, GaAs_die_error_norms.max() * 3])

ax52 = inset_axes(ax5, width="55%", height="50%")
ax52.scatter(
    GaAs_die_true_norms,
    GaAs_die_pred_norms,
    c=cm.haline_r(GaAs_die_relative_errors / die_max),
)
ax52.grid(linestyle="-.")
ax52.plot(
    [GaAs_die_true_norms.min(), GaAs_die_true_norms.max()],
    [GaAs_die_true_norms.min(), GaAs_die_true_norms.max()],
    "--",
    c="gray",
)
ax52.tick_params(axis="both", labelsize=10)

plt.show()
# plt.savefig("fig.pdf")
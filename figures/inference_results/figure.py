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

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(
    nrows=5,
    ncols=8,
    width_ratios=[0.01, 1, 0.15, 0.7, 0.15, 0.7, 0.1, 0.05],
    height_ratios=[0.1, 1, 0.3, 1, 0.06],
    hspace=0,
    wspace=0,
    left=0.05,
    #    right=0.95,
    #    top=1,
    #    bottom=0.04,
)  # , left=0.06, right=0.90, hspace=0.05, wspace=0.20)

fig.text(0.01, 0.93, "a)", fontsize=30)
ax1 = fig.add_subplot(gs[1:4, 1])
ax1.grid(linestyle="-.")
ax1.set_ylabel("Test MAE")
ax1.set_xlabel("Dataset size")


cbax1 = plt.subplot(gs[1, 7])
cbax2 = plt.subplot(gs[3, 7])

ax2 = fig.add_subplot(gs[1, 3])
ax2.grid(linestyle="-.")

true_norms = np.linalg.norm(BN_pol_trues, 2, axis=-1)
pred_norms = np.linalg.norm(BN_pol_preds, 2, axis=-1)
errors = BN_pol_preds - BN_pol_trues
error_norms = np.linalg.norm(errors, 2, axis=-1)
relative_errors = error_norms / true_norms

im2 = ax2.scatter(
    true_norms,
    error_norms,
    c=cm.haline_r(relative_errors / (relative_errors.max() - relative_errors.min())),
)
im2.set_cmap(cm.haline_r)
im2.set_clim(relative_errors.min(), relative_errors.max())
cbar = plt.colorbar(im2, cax=cbax1, use_gridspec=True, format="%.1e")
cbar.set_label("Relative error")
ax2.set_ylim([0, error_norms.max() * 3])
ax2.set_xlabel(r"$|\bm{P}_{true}| [me/a^{2}_0]$")
ax2.set_ylabel(r"$|\bm{P}_{true} - \bm{P}_{pred}| [me/a^{2}_0]$")
ax2.set_title("BN", fontsize=30)

ax22 = inset_axes(ax2, width="55%", height="50%")
ax22.scatter(
    true_norms,
    pred_norms,
    c=cm.haline_r(relative_errors / (relative_errors.max() - relative_errors.min())),
)
ax22.grid(linestyle="-.")
# ax22.set_xlabel(r"True $[me/a^{2}_0]$", fontsize=12)
# ax22.set_ylabel(r"Predicted $[me/a^{2}_0]$", fontsize=12)
ax22.plot(
    [true_norms.min(), true_norms.max()],
    [true_norms.min(), true_norms.max()],
    "--",
    c="gray",
)
ax22.tick_params(axis="both", labelsize=10)

ax3 = fig.add_subplot(gs[1, 5])
ax3.grid(linestyle="-.")
ax3.set_xlabel(r"$|\bm{P}_{true}| [me/a^{2}_0]$")
ax3.set_title("GaAs", fontsize=30)

ax4 = fig.add_subplot(gs[3, 3])
ax4.grid(linestyle="-.")
ax4.set_xlabel(r"$|\epsilon^\infty_{true}| [\epsilon_0]$")
ax4.set_ylabel(r"$|\epsilon^\infty_{true} - \epsilon^\infty_{pred}| [\epsilon_0]$")

ax5 = fig.add_subplot(gs[3, 5])
ax5.grid(linestyle="-.")
ax5.set_xlabel(r"$|\epsilon^\infty_{true}| [\epsilon_0]$")


plt.show()
# plt.savefig("fig.pdf")

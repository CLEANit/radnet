import matplotlib.pyplot as plt
import numpy as np
import os

font = {"family": "CMU Serif", "size": 18}
plt.rc("font", **font)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

datadir = "data/"
BN_ml_data_path = os.path.join(datadir, "BN_ml_raman_spec.dat")
BN_dft_data_path = os.path.join(datadir, "BN_dft_raman_spec.dat")
GaAs_ml_data_path = os.path.join(datadir, "GaAs_ml_raman_spec.dat")
GaAs_dft_data_path = os.path.join(datadir, "GaAs_dft_raman_spec.dat")

BN_ml_data = np.loadtxt(BN_ml_data_path)
BN_dft_data = np.loadtxt(BN_dft_data_path)
GaAs_ml_data = np.loadtxt(GaAs_ml_data_path)
GaAs_dft_data = np.loadtxt(GaAs_dft_data_path)

BN_ml_intensities = BN_ml_data[:, 1] / np.sum(BN_ml_data[:, 1])
BN_dft_intensities = BN_dft_data[:, 1] / np.sum(BN_dft_data[:, 1])
GaAs_ml_intensities = GaAs_ml_data[:, 1] / np.sum(GaAs_ml_data[:, 1])
GaAs_dft_intensities = GaAs_dft_data[:, 1] / np.sum(GaAs_dft_data[:, 1])

line_color = "#b35900"
fill_color = "#59d97f"


fig = plt.figure(figsize=(7, 12))
gs = fig.add_gridspec(
    nrows=2,
    ncols=1,
    left=0.07,
    right=0.94,
    top=0.965,
    bottom=0.06,
    hspace=0.22,
)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("BN", fontsize=25)
ax1.set_ylim([0, 0.02])
ax1.set_xlim([800, 1400])
ax1.plot(
    BN_ml_data[:, 0],
    BN_ml_intensities,
    color=line_color,
    ls="-.",
    label="RADNET",
    linewidth=1.5,
)
ax1.fill_between(
    x=BN_dft_data[:, 0], y1=BN_dft_intensities, color=fill_color, label="DFT"
)
ax1.yaxis.set_ticklabels([])
ax1.set_ylabel("Relative Raman Intensity", fontsize=20)
ax1.set_xlabel(r"Frequency (cm$^{-1}$)", fontsize=20)
ax1.legend(fontsize=20)
ax1.grid("on")


ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title("GaAS", fontsize=25)
ax2.set_ylim([0, 0.03])
ax2.set_xlim([125, 380])
ax2.plot(
    GaAs_ml_data[:, 0], GaAs_ml_intensities, color=line_color, ls="-.", label="RADNET"
)
ax2.fill_between(
    x=GaAs_dft_data[:, 0], y1=GaAs_dft_intensities, color=fill_color, label="DFT"
)
ax2.yaxis.set_ticklabels([])
ax2.set_ylabel("Relative Raman Intensity", fontsize=20)
ax2.set_xlabel(r"Frequency (cm$^{-1}$)", fontsize=20)
ax2.legend(fontsize=20)
ax2.grid("on")


# plt.show()
plt.savefig("figure_spectrum.eps")

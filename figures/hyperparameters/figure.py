import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


data_dir = "data"
inference_pol_rcut_path = os.path.join(data_dir, "BN_pol_inference_1.5_4.0.pkl")
inference_die_rcut_path = os.path.join(data_dir, "BN_die_inference_1.5_4.0.pkl")
rcut_effch_data_path = os.path.join(data_dir, "BN_effch_1.5_4.0.npy")
rcut_suscept_data_path = os.path.join(data_dir, "BN_suscept_1.5_4.0.npy")

rcut_effch_data = np.load(rcut_effch_data_path)
rcut_suscept_data = np.load(rcut_suscept_data_path)
with open(inference_pol_rcut_path, "rb") as f:
    inf_pol_rcut_data = pickle.load(f)
with open(inference_die_rcut_path, "rb") as f:
    inf_die_rcut_data = pickle.load(f)


inf_pol_rcut_means = np.mean(inf_pol_rcut_data["maes"], axis=1)
inf_pol_rcut_stds = np.std(inf_pol_rcut_data["maes"], axis=1)
inf_die_rcut_means = np.mean(inf_die_rcut_data["maes"], axis=1)
inf_die_rcut_stds = np.std(inf_die_rcut_data["maes"], axis=1)

rcut_effch_data = rcut_effch_data.reshape(
    rcut_effch_data.shape[0], rcut_effch_data.shape[1], -1
)[1:]
rcut_effch_stds = np.mean(np.std(rcut_effch_data, axis=1), axis=1)
rcut_suscept_data = rcut_suscept_data.reshape(
    rcut_suscept_data.shape[0], rcut_suscept_data.shape[1], -1
)[1:]
rcut_suscept_stds = np.mean(np.std(rcut_suscept_data, axis=1), axis=1)


rcuts = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

font = {"family": "CMU Serif", "size": 18}
plt.rc("font", **font)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")
plt.rcParams["axes.formatter.limits"] = (0, 0)

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(nrows=2, ncols=2, left=0.06, right=0.90, hspace=0.0, wspace=0.20)

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Cutoff convergence", fontsize=25)
pol1 = ax1.errorbar(rcuts, inf_pol_rcut_means, yerr=inf_pol_rcut_stds, color="b")
ax1.grid(linestyle="-.")
ax1.set_ylabel("Test MAE")
ax1.set_xlabel("Cutoff radius (\AA)")
ax1.set_yscale("log")
ax1.spines["left"].set_edgecolor(pol1[0].get_color())
ax1.tick_params(axis="y", which="both", colors=pol1[0].get_color())
ax2 = ax1.twinx()
die2 = ax2.errorbar(rcuts, inf_die_rcut_means, yerr=inf_die_rcut_stds, color="r")
ax2.set_yscale("log")
ax2.spines["right"].set_edgecolor(die2[0].get_color())
ax2.tick_params(axis="y", which="both", colors=die2[0].get_color())

ax1.legend([pol1, die2], ["Polarization", "Dielectric"])


ax3 = fig.add_subplot(gs[0, 1])
ax3.grid(linestyle="-.")
ax3.set_ylim([0, None])
ax3.yaxis.tick_right()

ax4 = ax3.twinx()
ax4.set_ylabel("Test MAE")
ax4.set_xlim(ax3.get_xlim())

ax5 = fig.add_subplot(gs[1, 0])
eff5 = ax5.plot(
    rcuts[1:], rcut_effch_stds, label="Effective charges (e)", color=pol1[0].get_color()
)
ax5.grid(linestyle="-.")
ax5.set_ylim([0, None])
ax5.set_xlabel("Cutoff radius (\AA)")
ax5.set_ylabel("Derivatives STD")
ax5.set_xlim(ax1.get_xlim())
ax5.spines["left"].set_edgecolor(eff5[0].get_color())
ax5.tick_params(axis="y", which="both", colors=eff5[0].get_color())
ax5.yaxis.set_ticks([0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
ax5.yaxis.set_ticklabels(["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5"])
fig.text(
    ax5.get_position().get_points()[0, 0] - 0.04,
    ax5.get_position().get_points()[1, 1] - 0.015,
    r"$\times 10^{-2}$",
    color=eff5[0].get_color(),
)


ax6 = ax5.twinx()
sus6 = ax6.plot(
    rcuts[1:],
    rcut_suscept_stds,
    color=die2[0].get_color(),
)
ax6.set_ylim([0, None])
ax6.yaxis.set_ticks([0, 0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175])
ax6.yaxis.set_ticklabels(["0.0", "2.5", "5.0", "7.5", "10.0", "12.5", "15.0", "17.5"])
fig.text(
    ax6.get_position().get_points()[1, 0] + 0.005,
    ax6.get_position().get_points()[1, 1] - 0.015,
    r"$\times 10^{-4}$",
    color=sus6[0].get_color(),
)
ax6.spines["right"].set_edgecolor(sus6[0].get_color())
ax6.tick_params(axis="y", which="both", colors=sus6[0].get_color())
ax6.yaxis.label.set_color(sus6[0].get_color())
ax5.legend(
    [eff5[0], sus6[0]],
    [r"$\Delta Z^{*}$", r"$\Delta\frac{\partial \chi}{\partial R}$"],
    loc="lower left",
)

ax7 = fig.add_subplot(gs[1, 1])
ax7.grid(linestyle="-.")
ax7.set_ylim([0, None])
ax7.set_xlabel("Image Shape")

ax8 = ax7.twinx()
ax8.set_xlabel(ax7.get_xlabel())
ax8.set_ylabel("Derivatives STD")

plt.show()
# plt.savefig("fig.pdf")

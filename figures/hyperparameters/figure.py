import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


data_dir = "data"
inference_pol_rcut_path = os.path.join(data_dir, "BN_pol_inference_1.5_4.0.pkl")
effch_data_path = os.path.join(data_dir, "BN_effch_1.5_4.0.npy")

effch_data = np.load(effch_data_path)
with open(inference_pol_rcut_path, "rb") as f:
    inf_pol_rcut_data = pickle.load(f)


inf_pol_rcut_means = np.mean(inf_pol_rcut_data["maes"][1:], axis=1)
inf_pol_rcut_stds = np.std(inf_pol_rcut_data["maes"][1:], axis=1)

effch_data = effch_data.reshape(effch_data.shape[0], effch_data.shape[1], -1)[1:]
effch_stds = np.mean(np.std(effch_data, axis=1), axis=1)




rcuts = [2.0, 2.5, 3.0, 3.5, 4.0]

font = {"family": "CMU Serif", "size": 18}
plt.rc("font", **font)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(nrows=2, ncols=2, left=0.06, right=0.90, hspace=0.0, wspace=0.20)
#fig, ax = plt.subplots(nrows=2, ncols=2)
#(ax1, ax2), (ax3, ax4) = ax

ax1 = fig.add_subplot(gs[0,0])
pol1 = ax1.errorbar(rcuts, inf_pol_rcut_means, yerr=inf_pol_rcut_stds, color="b")
ax1.grid(linestyle="-.")
ax1.set_ylim([0, None])
ax1.set_ylabel("Test MAE")
ax1.set_xlabel("Cutoff radius (\AA)")
ax2 = ax1.twinx()
die2 = ax2.errorbar(rcuts, np.linspace(0,1, len(rcuts)), yerr=0.1, color="r")

#print(type(pol1[0]), type(die2[0]))

ax1.legend([pol1, die2], ["Polarization", "Dielectric"])

ax3 = fig.add_subplot(gs[0,1])
ax3.grid(linestyle="-.")
ax3.set_ylim([0, None])
ax3.set_ylabel("Test MAE")
ax3.yaxis.set_label_position("right")
ax3.yaxis.tick_right()

ax5 = fig.add_subplot(gs[1,0])
ax5.plot(rcuts, effch_stds, label="Effective charges (e)")
ax5.grid(linestyle="-.")
ax5.set_ylim([0, None])
ax5.set_xlabel("Cutoff radius (\AA)")
ax5.set_ylabel("Derivatives STD (e)")

ax6 = fig.add_subplot(gs[1,1])
ax6.grid(linestyle="-.")
ax6.set_ylim([0, None])
ax6.set_xlabel("Image Shape")

plt.show()
#plt.savefig("fig.pdf")

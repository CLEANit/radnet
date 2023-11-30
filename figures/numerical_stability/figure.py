import matplotlib.pyplot as plt
import numpy as np
import os


data_dir = "data"
pol_data_path = os.path.join(data_dir, "BN_pol_test.npy")
effch_data_path = os.path.join(data_dir, "BN_effch_test.npy")

data = np.load(effch_data_path)
data = data.reshape(data.shape[0], data.shape[1], -1)

stds = np.mean(np.std(data, axis=1), axis=1)
rcuts = [2.0, 4.0]

# font = {"family": "CMU Serif", "size": 18}
# plt.rc("font", **font)
# plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{bm}")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(rcuts, stds)

plt.savefig("fig.pdf")

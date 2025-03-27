from matplotlib import pyplot as plt
import numpy as np
import csv
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--paths', type=str, action='append', required=True, help="Add a path to a sampling test file")
parser.add_argument('-s', '--sampling_methods', type=str, action='append', required=True, choices=['uniform', 'cluster', 'empirical'], help="Add a sampling method (in order)")
parser.add_argument('-o', '--out', type=str, required=True, help="Path to directory where output files are stored.")
args = parser.parse_args()

sample_to_errors = {}
sample_to_mapes = {}
files = args.paths
methods = args.sampling_methods
colors = ["#648FFF", "#DC267F", "#FFB000"][:len(methods)]
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
figs = [fig1, fig2]
axs = [ax1, ax2]
for method_i in range(len(methods)):
    passed_mses = False
    errors = []
    mapes = []
    min_samples = 20
    with open(files[method_i], 'r') as f:
        reader = csv.reader(f)
        for i in range(8):
            next(reader, None)
        l = 0
        for j, row in enumerate(reader):
            if "MAPEs" in row[0]:
                passed_mses = True
                continue

            values = [float(i) for i in row]
            if j == 0:
                l = len(row)
            if not passed_mses:
                errors.append(values[0:l])
            else:
                mapes.append(values[0:l])
        
        errors = np.asarray(errors)
        mapes = np.asarray(mapes)

    sample_to_errors[methods[method_i]] = errors
    sample_to_mapes[methods[method_i]] = mapes

    xs = np.array([i + min_samples for i in range(len(errors[0]))])
    mse_means = np.mean(errors, axis=0)
    mse_mins = np.min(errors, axis=0)
    mse_maxs = np.max(errors, axis=0)
    mape_means = np.mean(mapes, axis=0)
    mape_mins = np.min(mapes, axis=0)
    mape_maxs = np.max(mapes, axis=0)
    axs[0].plot(xs, mse_means, label = f"{methods[method_i]} mean MSE", color=colors[method_i], linewidth=0.6)
    axs[0].fill_between(xs, mse_mins, mse_maxs, color=colors[method_i], alpha=0.2, label=f"Min-Max region for {methods[method_i]}")
    axs[1].plot(xs, mape_means, label = f"{methods[method_i]} mean MAPE", color=colors[method_i], linewidth=0.6)
    axs[1].fill_between(xs, mape_mins, mape_maxs, color=colors[method_i], alpha=0.2, label=f"Min-Max region for {methods[method_i]}")
    # Plotting arbitrary baseline for comparison
    axs[1].plot(xs, [0.25] * len(xs), linestyle='--', color='black', label="__nolegend__", linewidth=0.5)


axs[0].set_ylim([0, 3])
axs[0].set_xlabel("# of samples (n)")
axs[0].set_ylabel("Mean squared error (MSE)")
axs[0].legend()
axs[0].set_title("MSE vs n for different sampling methods")
axs[1].set_ylim([0, 1])
axs[1].set_xlabel("# of samples (n)")
axs[1].set_ylabel("Mean absolute percentage error (MAPE)")
axs[1].legend()
axs[1].set_title("MAPE vs n for different sampling methods")

figs[0].savefig(os.path.join(args.out, "MSE_together.png"))
figs[1].savefig(os.path.join(args.out, "MAPE_together.png"))
# plt.show()
# plt.savefig("./files/sampling_tests/put_together.png")
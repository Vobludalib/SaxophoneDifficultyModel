from matplotlib import pyplot as plt
import numpy as np
import csv

sample_to_errors = {}
files = ["./files/sampling_tests/test_3958758.csv", "./files/sampling_tests/test_3160455.csv", "./files/sampling_tests/test_984637.csv"]
methods = ["Empirical", "Uniform", "Cluster"]
colors = ["#648FFF", "#DC267F", "#FFB000"]
for method_i in range(3):
    errors = []
    min_samples = 20
    with open(files[method_i], 'r') as f:
        reader = csv.reader(f)
        for i in range(6):
            next(reader, None)
        l = 0
        for j, row in enumerate(reader):
            values = [float(i) for i in row]
            if j == 0:
                l = len(row)
            errors.append(values[0:l])
        
        errors = np.asarray(errors)

    sample_to_errors[methods[method_i]] = errors

    xs = np.array([i + min_samples for i in range(len(errors[0]))])
    means = np.mean(errors, axis=0)
    mins = np.min(errors, axis=0)
    maxs = np.max(errors, axis=0)
    plt.plot(xs, means, label = f"{methods[method_i]} mean MSE", color=colors[method_i])
    plt.fill_between(xs, mins, maxs, color=colors[method_i], alpha=0.2, label=f"Min-Max region for {methods[method_i]}")

ax = plt.gca()
ax.set_ylim([0, 3])
plt.xlabel("# of samples (n)")
plt.ylabel("Mean squared error (MSE)")
plt.legend()
plt.title("MSE vs n for different sampling methods")

# plt.show()
plt.savefig("./files/sampling_tests/put_together.png")
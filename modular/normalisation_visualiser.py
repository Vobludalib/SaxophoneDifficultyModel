import encoding
import csv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch

raw_dict = {}
with open('/Users/slibricky/Desktop/Thesis/thesis/modular/data.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for line in reader:
        name = line[0]
        trill_speed = float(line[-1])
        if trill_speed == 0:
            continue
        raw_dict[name] = trill_speed

normalised_dict = {}
with open('/Users/slibricky/Desktop/Thesis/thesis/modular/files/normalisation_csvs/ALL_DATA.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for line in reader:
        name = line[0]
        trill_speed = float(line[-1])
        if name in raw_dict:
            normalised_dict[name] = trill_speed

session_start_markers = [0, 64, 127, 185, 252, 313, 313+66, 313+66+72]
xs = []
ys = []
colors = []
arrow_tuples = []

def isolate_number(str):
    prefix, numberstr = str.split("Trill")
    numberstr, ext = numberstr.split(".")
    return(int(numberstr))

sorted_keys = sorted(list(normalised_dict.keys()), key=isolate_number)
for i, filename in enumerate(sorted_keys):
    xs.append(i)
    ys.append(raw_dict[filename])
    colors.append('blue')
    xs.append(i)
    ys.append(normalised_dict[filename])
    colors.append('red')
    arrow_tuples.append((i, raw_dict[filename], normalised_dict[filename]))

plt.scatter(xs, ys, color=colors)
plt.xlabel("Individual transitions in session order")
plt.ylabel("Trill speeds in trills/s")
legend_handles = [
    Patch(facecolor="blue", label="Raw values"),
    Patch(facecolor="red", label="Post-normalisation")
                  ]
ax = plt.gca()
fig = plt.gcf()
fig.set_size_inches(8, 6)
ax.legend(handles=legend_handles, loc="upper right")
ax.set_xticks(session_start_markers)
ax.set_xticklabels([f"Session {i}" for i in range(len(session_start_markers))])
plt.grid(axis="x", linestyle="--", linewidth=0.5)
plt.title("Visualisation of normalisation per session, lambda=0.2")
for i, raw, norm in arrow_tuples:
    plt.annotate('', xy=(i, norm), xycoords='data', xytext=(i, raw), textcoords='data', arrowprops=dict(facecolor='black', arrowstyle='->'))
# plt.show()
plt.savefig("./normalisation_vis.png")
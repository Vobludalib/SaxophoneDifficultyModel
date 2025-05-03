import encoding
import csv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True, help="Path to processed data .csv file.")
parser.add_argument('-i', '--input_norm_file', type=str, required=True, help="Path to the normalised .csv file.")
parser.add_argument('-o', '--out', type=str, default=os.path.join(".", "normalisation_vis.png"), required=False)
parser.add_argument('-n', '--norm_style', type=str, required=True, help="Text that describes the normalisation style")
parser.add_argument('-s', '--strength', type=str, required=True, help="Normalisation strength text value")
args = parser.parse_args()

raw_dict = {}
name_to_session = {}
session_to_player = {}
with open(args.data, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for line in reader:
        name = line[0]
        player = line[2]
        session = int(line[3])
        trill_speed = float(line[-1])
        if trill_speed == 0:
            continue

        if raw_dict.get(session, None) is None:
            raw_dict[session] = {name: trill_speed}
        else:
            raw_dict[session][name] = trill_speed

        name_to_session[name] = session
        session_to_player[session] = player

normalised_dict = {}
with open(args.input_norm_file, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for line in reader:
        name = line[0]
        trill_speed = float(line[-1])
        if name in name_to_session:
            session = name_to_session[name]
            if normalised_dict.get(session, None) is None:
                normalised_dict[session] = {name: trill_speed}
            else:
                normalised_dict[session][name] = trill_speed

session_delimiters = [0]
for session in raw_dict:
    session_delimiters.append(session_delimiters[-1] + len(raw_dict[session]))

xs = []
ys = []
colors = []
arrow_tuples = []

for session_i, session in enumerate(raw_dict):
    for i, filename in enumerate(raw_dict[session]):
        scaled_x = session_delimiters[session_i] + 10 + (i / (session_delimiters[session_i + 1] - session_delimiters[session_i])) * (session_delimiters[session_i] - session_delimiters[session_i + 1] - 20)
        xs.append(scaled_x)
        ys.append(raw_dict[session][filename])
        colors.append('#648FFF')
        xs.append(scaled_x)
        ys.append(normalised_dict[session][filename])
        colors.append('#FE6100')
        arrow_tuples.append((scaled_x, raw_dict[session][filename], normalised_dict[session][filename]))

plt.scatter(xs, ys, color=colors)
plt.xlabel("Individual transitions in session order")
plt.ylabel("Trill speeds in trills/s")
legend_handles = [
    Patch(facecolor="#648FFF", label="Raw values"),
    Patch(facecolor="#FE6100", label="Post-normalisation")
                  ]
ax = plt.gca()
fig = plt.gcf()
fig.set_size_inches(8, 6)
ax.legend(handles=legend_handles, loc="upper center")
ax.set_xticks(session_delimiters)
ax.set_xticklabels([f"{'   ' if i < 12 else ''}S{i+1}" for i in range(len(session_delimiters) - 1)] + [''])
ax.set_xlim(session_delimiters[0], session_delimiters[-1])
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("left")
plt.grid(axis="x", linestyle="--", linewidth=0.5)
plt.title(f"Visualisation of {args.norm_style} normalisation per session, lambda={args.strength}")
for i, raw, norm in arrow_tuples:
    plt.annotate('', xy=(i, norm), xycoords='data', xytext=(i, raw), textcoords='data', arrowprops=dict(facecolor='black', arrowstyle='->'))
# plt.show()
plt.savefig(args.out)
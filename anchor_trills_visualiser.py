import encoding
import normalisation_tool
import csv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import os
import numpy as np
import random

directory = "./files/data_processed/"
file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

anchors_file = "./documentation/anchor_transitions.txt"
anchor_encoding_pairs = []
with open(anchors_file, 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        encoding1 = line[0]
        encoding2 = line[1]
        anchor_encoding_pairs.append((encoding1, encoding2))

anchors_from_dir = normalisation_tool.load_anchors_from_directory("./files/data_processed")

player_to_anchor_trills = {}

for session in anchors_from_dir.keys():
    player_name = session.split("Session")[0]
    if player_to_anchor_trills.get(player_name, None) is None:
        player_to_anchor_trills[player_name] = [anchors_from_dir[session]]
    else:
        player_to_anchor_trills[player_name].append(anchors_from_dir[session])

vals = {}
player_colors = []
player_order = []
player_colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#ffcc33']

for player in player_to_anchor_trills.keys():
    sessions = player_to_anchor_trills[player]
    for session in sessions:
        for fing1, fing2, speed in session:
            key = (fing1.generate_encoding(), fing2.generate_encoding())
            x = anchor_encoding_pairs.index(key)
            # x_names[x] = f"{fing1.__str__()[0:3]}, {fing2.__str__()[0:3]}"
            if vals.get(x, None) is None:
                vals[x] = [(speed, player)]
            else:
                vals[x].append((speed, player))

    player_order.append(player)

x_names = [f"    Trans. {i+1}" for i in range(len(player_order)+1)]

xs = []
ys = []
colors = []
for x in vals.keys():
    in_order = sorted(vals[x], key=lambda speed_player: speed_player[0])
    for i, (speed, player) in enumerate(in_order):
        xs.append(x + 0.2 + i*0.05)
        ys.append(speed)
        player_index = player_order.index(player)
        colors.append(player_colors[player_index])

player_order = [f"Player {i+1}" for i in range(len(player_order))]

ax = plt.gca()
plt.scatter(xs, ys, color=colors)
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=x_names)
plt.xlim((0, 6))
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("left")
plt.xlabel("Individual repeated transitions")
plt.ylabel("Trill speed in trills/s")
plt.title("Visualisation of trill speed variance of repeated transitions")
plt.grid(visible=True, axis='x')
legend_patches = [Patch(color=color, label=name) for color, name in zip(player_colors, player_order)]
plt.legend(handles=legend_patches, loc="upper center")
plt.ylim((0, 8))
# plt.show()
plt.savefig("variance_vis.png")

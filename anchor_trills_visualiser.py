import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import argparse
import statistics
import sklearn
import sklearn.metrics

anchors_file = os.path.join(".", "encodings", "anchor_transitions.txt")
anchor_encoding_pairs = []
with open(anchors_file, 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        encoding1 = line[0]
        encoding2 = line[1]
        anchor_encoding_pairs.append((encoding1, encoding2))

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, required=True, help="Path to processed data .csv file.")
parser.add_argument('-o', '--out', type=str, default=os.path.join(".", "variance_vis.png"), required=False)
args = parser.parse_args()

player_to_anchor_trills = {}
with open(args.data, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if row[1] == "-1":
            player = row[2]
            session = int(row[3])
            encoding1 = row[6]
            encoding2 = row[9]
            ts = float(row[10])
            if player_to_anchor_trills.get(player, None) is None:
                player_to_anchor_trills[player] = {session: [(encoding1, encoding2, ts)]}
            else:
                if player_to_anchor_trills[player].get(session, None) is None:
                    player_to_anchor_trills[player][session] = [(encoding1, encoding2, ts)]
                else:
                    player_to_anchor_trills[player][session].append((encoding1, encoding2, ts))

vals = {}
player_colors = []
player_order = []
player_colors = ['#ffcc33', '#ff7f00', '#4daf4a', '#f781bf', '#377eb8']

for player in player_to_anchor_trills.keys():
    sessions = player_to_anchor_trills[player].keys()
    for session in sessions:
        for fing1, fing2, speed in player_to_anchor_trills[player][session]:
            key = (fing1, fing2)
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

mape_errors = []
for trans_val in vals.keys():
    speeds = []
    for speed, _ in vals[trans_val]:
        speeds.append(speed)
    median = statistics.median(speeds)
    y_trues = [median for i in range(len(speeds))]
    mape_error = sklearn.metrics.mean_absolute_percentage_error(y_trues, speeds)
    mape_errors.append(mape_error)
    print(f"Trans {trans_val}: {mape_error}")
    print(anchor_encoding_pairs[trans_val])

print(f"Avg MAPE: {sum(mape_errors)/len(mape_errors)}, Min: {min(mape_errors)}, Max: {max(mape_errors)}")

ax = plt.gca()
plt.scatter(xs, ys, color=colors)
plt.xticks(ticks=[0, 1, 2, 3, 4, 5], labels=x_names)
plt.xlim((0, 6))
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("left")
plt.xlabel("Individual anchor transitions")
plt.ylabel("Trill speed in trills/s")
plt.title("Visualisation of trill speed variance of anchor transitions")
plt.grid(visible=True, axis='x')
legend_patches = [Patch(color=color, label=name) for color, name in zip(player_colors, player_order)]
plt.legend(handles=legend_patches, loc="upper center")
plt.ylim((0, 8))
# plt.show()
plt.savefig(args.out)

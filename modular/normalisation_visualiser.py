import encoding
import csv
import matplotlib.pyplot as plt

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
with open('/Users/slibricky/Desktop/Thesis/thesis/modular/files/normalisation_csvs/session0to3.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for line in reader:
        name = line[0]
        trill_speed = float(line[-1])
        if name in raw_dict:
            normalised_dict[name] = trill_speed

xs = []
ys = []
colors = []
arrow_tuples = []
for i, filename in enumerate(normalised_dict.keys()):
    xs.append(i)
    ys.append(raw_dict[filename])
    colors.append('blue')
    xs.append(i)
    ys.append(normalised_dict[filename])
    colors.append('red')
    arrow_tuples.append((i, raw_dict[filename], normalised_dict[filename]))

plt.scatter(xs, ys, color=colors)
for i, raw, norm in arrow_tuples:
    plt.annotate('', xy=(i, norm), xycoords='data', xytext=(i, raw), textcoords='data', arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.show()
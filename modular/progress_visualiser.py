import matplotlib.pyplot as plt
import encoding
import itertools
import numpy as np

data = encoding.load_transitions_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/files/PlatonSession0BatchAnalysed.csv")
all_fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")

combs = encoding.generate_all_transitions(all_fingerings)
plot_data = []
for possible_transition in combs:
    trans = encoding.Transition(possible_transition[0], possible_transition[1])
    if trans in data:
        plot_data.append(1)
    else:
        plot_data.append(0)

pdlen = len(plot_data)
completion = (np.sum(plot_data) / pdlen) * 100
for i in range(29*26 - pdlen):
    plot_data.append(np.nan)
plot_data = np.asarray(plot_data)

cmap = plt.cm.Blues
cmap.set_bad(color='red')  # Set NaN areas to white

plt.title(f"Progress update on data collection -> {completion:.1f}%")
plt.pcolormesh(plot_data.reshape(26, 29), cmap=cmap, edgecolors='gray', linewidth=0.5)
plt.show()

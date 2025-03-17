import encoding
import matplotlib.pyplot as plt
import numpy as np

trans_to_trill_dict = encoding.load_transitions_from_file('./files/normalisation_csvs/ALL_DATA.csv')
xss = list(trans_to_trill_dict.values())
speeds = np.asarray([ x for xs in xss for x in xs]).T

plt.hist(speeds, bins=20)
plt.show()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import numpy as np
import random

def main():
    with open("/Users/slibricky/Desktop/Thesis/thesis/modular/files/BatchAnalysed.csv", 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        cluster_dict = {}
        for line in reader:
            cluster = int(line[1])
            speed = float(line[8])
            if cluster not in cluster_dict:
                cluster_dict[cluster] = [speed]
            else:
                cluster_dict[cluster].append(speed)

        xs = flat_list = [
                            speed
                            for cluster in cluster_dict.values()
                            for speed in cluster
                        ]
        ys_prep = [(len(cluster_dict[key])) for key in cluster_dict.keys()]
        print(ys_prep)
        ys = []
        colors = []
        for label, cluster_len in enumerate(ys_prep):
            cluster_color = (random.uniform(0,1), random.uniform(0, 1), random.uniform(0, 1))
            for i in range(cluster_len): 
                ys.append(label)
                colors.append(cluster_color)
        plt.scatter(xs, ys, color=colors)
        plt.show()

if __name__ == "__main__":
    main()
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

        xs = [
                            speed
                            for key in cluster_dict.keys()
                            for speed in cluster_dict[key] if key != -1
                        ]
        logxs = np.log2(xs)
        ys_prep = [(len(cluster_dict[key]), key) for key in cluster_dict.keys() if key != -1]
        ys = []
        colors = []
        annotations = []
        log_annotations = []
        for label, (cluster_len, key) in enumerate(ys_prep):
            cluster_color = (random.uniform(0,1), random.uniform(0, 1), random.uniform(0, 1))
            max_speed_in_cluster = np.max(cluster_dict[key])
            annotations.append((label, key, max_speed_in_cluster))
            log_annotations.append((label, key, np.log2(max_speed_in_cluster)))
            for i in range(cluster_len): 
                ys.append(label)
                colors.append(cluster_color)
        plt.scatter(xs, ys, color=colors)
        for (y, cluster_id, max) in annotations:
            plt.annotate(str(cluster_id), (max, y))
        plt.savefig('./files/session_clusters_normal.png')
        plt.clf()
        plt.scatter(logxs, ys, color=colors)
        for (y, cluster_id, max) in log_annotations:
            plt.annotate(str(cluster_id), (max, y))
        plt.savefig('./files/log_session_clusters_normal.png')

if __name__ == "__main__":
    main()
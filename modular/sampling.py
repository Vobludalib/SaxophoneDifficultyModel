import itertools
import encoding
import numpy as np
import sklearn.cluster as skc
import random

def generate_all_transitions(fingerings):
    all_transitions = itertools.combinations(fingerings, 2)
    return all_transitions

# FROM ALL FINGERINGS SUBSAMPLE USING 
# a) UNIFORM RANDOMNESS
# b) FREQUENCY IN WJDB
# c) BASED ON CLUSTERING

# Method for subsampling using uniform randomness

# Method for subsampling using WJDB frequencies

# TODO: TEST SET created from leftovers from sampling -> subsample leftovers to get consistent test set size
# Would mean we can only subsample within some range for testing (85% to 15%)
# Is it possible to reserve some data as a test set?
# Cons for me: We don't have a fixed size for each cluster, so we cannot just take one from each cluster or something like that
# Method for subsampling based on clusters
def cluster_subsample(transitions, n, seed=10, print_debug=False):
    random.seed(seed)

    encoding_feature_pairs = []
    for transition in transitions:
        if print_debug: print(f"Going from {transition[0].name} to {transition[1].name}")
        name, features = encoding.generate_interval_features(transition)
        encoding_feature_pairs.append(([transition[0], transition[1]], features))
    
    clusters_dict = {}
    _, labels, _ = skc.k_means(n_clusters=n, X=np.asarray([pair[1] for pair in encoding_feature_pairs]))
    for label in range(n):
        if print_debug: print(f"=== PROCESSING LABEL {label} ===")
        for index, elem in enumerate(encoding_feature_pairs):
            if labels[index] == label:
                if print_debug: print(f"{elem[0]} with feat {elem[1]}")
                if label not in clusters_dict:
                    clusters_dict[label] = [elem[0]]
                else:
                    clusters_dict[label].append(elem[0])

    if print_debug: print(f"In total there are {len(encoding_feature_pairs)} unique trills using default fingerings on TS")

    selected_transitions = []
    not_selected_transition = []
    for cluster in clusters_dict:
        selected_transitions.append(random.sample(clusters_dict[cluster], 1)[0])
        for transition in clusters_dict[cluster]:
            if transition is not selected_transitions[-1]:
                not_selected_transition.append(transition)

    return selected_transitions, not_selected_transition

def main():
    fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")
    all_transitions = generate_all_transitions(fingerings)
    selected, _ = cluster_subsample(all_transitions, 50)
    for trans in selected:
        print(f"{trans[0].name} to {trans[1].name}")

if __name__ == '__main__':
    main()
import itertools
import encoding
import numpy as np
import sklearn.cluster as skc
import random
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_all_transitions(fingerings):
    all_transitions = itertools.combinations(fingerings, 2)
    return all_transitions

# FROM ALL FINGERINGS SUBSAMPLE USING 
# a) UNIFORM RANDOMNESS
# b) FREQUENCY IN WJDB
# c) BASED ON CLUSTERING

# Method for subsampling using uniform randomness
def uniform_subsample(transitions, n, seed=10, print_debug=False):
    random.seed(seed)
    sampled = random.sample(transitions, n)
    not_sampled = []
    for transition in transitions:
        if transition not in sampled:
            not_sampled.append(transition)

    return sampled, not_sampled

# Method for subsampling using WJDB frequencies
def occurence_frequency_subsample(transitions, n, seed=10, print_debug=False):

    transitions = list(transitions)

    pathToCsv = "/Users/slibricky/Desktop/Thesis/melospy-gui_V_1_4b_mac_osx/bin/analysis/feature+viz/bigramsTS.csv"

    bigramDict = { }

    # +14 here is transposing pitch -> Encoded MIDI values are as written, WJDB has MIDI as sounding

    def parseBigramsIntoDict(d: dict, bigramString):
        bigrams = str.split(bigramString, ':')
        for bigram in bigrams:
            midiVals = bigram.split(',')
            key = (0, 0)
            if midiVals[0] < midiVals[1]:
                key = (int(midiVals[0]) + 14, int(midiVals[1]) + 14)
            elif midiVals[0] == midiVals[1]:
                # Ignoring staying on the same note
                pass
            else:
                key = (int(midiVals[1]) + 14, int(midiVals[0]) + 14)

            d[key] = d.get(key, 0) + 1

    with open(pathToCsv, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=";")
        header = next(csvReader, None)
        for row in csvReader:
            parseBigramsIntoDict(bigramDict, row[1])

    midiToOccurencesDict = dict(sorted(bigramDict.items(), key= lambda x: x[1]))

    print(midiToOccurencesDict)

    midiToTransitionsDict = {}

    for transition in transitions:
        key = (transition[0].midi, transition[1].midi)
        if midiToTransitionsDict.get(key, None) is None:
            midiToTransitionsDict[key] = [transition]
        else:
            midiToTransitionsDict[key].append(transition)

    transitionsList = []
    occurencesList = []
    for transition in transitions:
        transitionsList.append(transition)
        key = (transition[0].midi, transition[1].midi)
        # Smoothing for unrepresented values
        occurencesList.append(midiToOccurencesDict.get(key, 0.01))

    probabilities = np.asarray(occurencesList) / np.sum(occurencesList)
    np.random.seed(seed)
    indexes = np.asarray([i for i in range(len(transitions))])
    selected_indexes = np.random.choice(a=indexes, p=probabilities, size=n, replace=False)
    selected = []
    for index in selected_indexes:
        selected.append(transitionsList[index])
    not_selected = []
    for transition in transitions:
        if transition not in selected:
            not_selected.append(transition)

    return selected, not_selected

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

# Using bootstrap sampling, we take a list of transitions and create a k n-sized test set
def generate_test_sets_from_leftovers(transitions, size_of_one_iteration, amount_of_iterations, seed=10, print_debug=False):
    random.seed(seed)
    iterations = []
    for i in range(amount_of_iterations):
        iterations.append(random.choices(transitions, k = size_of_one_iteration))
    return iterations

# Method to pair generated samples with the data values from a file -> generating a list of pairs of fingerings and a numpy array of their recorded trill speeds
def pair_list_of_transitions_with_recorded_values(transitions, path_to_data, seed=10):
    pass

def main():
    fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")
    all_transitions = list(generate_all_transitions(fingerings))
    amount_of_fingerings = len(all_transitions)
    errors = []
    size_of_test_set = 20
    number_of_test_sets_per_i = 10
    minimum_amount_of_elements_to_record = 20
    for i in tqdm(range(amount_of_fingerings - size_of_test_set - minimum_amount_of_elements_to_record)):
        amount_to_sample = i + size_of_test_set
        amount_leftover = amount_of_fingerings - amount_to_sample
        selected_cluster, leftover_cluster = cluster_subsample(all_transitions, amount_to_sample, seed=i)
        test_sets = generate_test_sets_from_leftovers(leftover_cluster, size_of_test_set, number_of_test_sets_per_i, seed=i+10)
        # TRAIN MODEL ON SELECTED CLUSTER
        error_sum = 0
        for test_set in test_sets:
            # EVALUATE MODEL ON TEST SET
            error_sum += 1 / random.randrange(1, i + 10)
        
        errors.append((i, error_sum/number_of_test_sets_per_i))
    
    unzipped = list(zip(*errors))
    xs = unzipped[0]
    ys = unzipped[1]
    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    main()
import sklearn.metrics
import encoding
import numpy as np
import sklearn.cluster as skc
import random
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import model
import sklearn

# FROM ALL FINGERINGS SUBSAMPLE USING 
# a) UNIFORM RANDOMNESS
# b) FREQUENCY IN WJDB
# c) BASED ON CLUSTERING

# Method for subsampling using uniform randomness
def uniform_subsample(transitions_trill_speed_dict: dict, n, seed=10, print_debug=False):
    random.seed(seed)

    sampled_trans = []
    sampled_speeds = []
    not_sampled_trans = []
    not_sampled_speeds = []

    amount_of_transitions = len(transitions_trill_speed_dict)
    transitions = list(transitions_trill_speed_dict.keys())

    indices = [i for i in range(amount_of_transitions)]
    for i in range(n):
        raw_index = random.randint(0, amount_of_transitions - i - 1)
        sampled_index = indices[raw_index]
        trans = transitions[sampled_index]
        speeds_for_trans = transitions_trill_speed_dict[trans]
        amount_of_speeds = len(speeds_for_trans)
        sampled_speed_index = random.randint(0, amount_of_speeds-1)
        for index, val in enumerate(speeds_for_trans):
            if index == sampled_speed_index:
                sampled_trans.append(trans)
                sampled_speeds.append(val)
            else:
                not_sampled_trans.append(trans)
                not_sampled_speeds.append(val)
        
        del indices[raw_index]
    
    for not_selected_index in indices:
        trans = transitions[not_selected_index]
        for speed in transitions_trill_speed_dict[trans]:
            not_sampled_trans.append(trans)
            not_sampled_speeds.append(speed)

    return sampled_trans, sampled_speeds, not_sampled_trans, not_sampled_speeds

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
        features = encoding.generate_transition_features(transition)
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

def bootstrap_sample(xs, ys, n, seed=10):
    random.seed(seed)
    indices = [random.randrange(0, len(xs)) for i in range(n)]
    sampled_xs = []
    sampled_ys = []
    for index in indices:
        sampled_xs.append(xs[index])
        sampled_ys.append(ys[index])
    
    return sampled_xs, sampled_ys

def perform_sampling_test(transitions_trill_speed_dict, sampling_method, size_of_test_set = 15):
    errors = []
    number_of_retries_per_i = 10
    minimum_amount_of_elements_in_train_set = 20
    amount_of_fingerings_recorded = len(transitions_trill_speed_dict)
    for i in tqdm(range(amount_of_fingerings_recorded - size_of_test_set - minimum_amount_of_elements_in_train_set)):
        seed_i = random.randint(0, 100000)
        amount_to_sample = i + minimum_amount_of_elements_in_train_set
        amount_leftover = amount_of_fingerings_recorded - amount_to_sample
        mse_sum = 0
        for j in range(number_of_retries_per_i):
            selected_xs, selected_ys, not_selected_xs, not_selected_ys = uniform_subsample(transitions_trill_speed_dict, amount_to_sample, seed=seed_i)
            bootstrap_xs, boostrap_ys = bootstrap_sample(not_selected_xs, not_selected_ys, size_of_test_set, seed=seed_i+j)
            selected_xs, selected_ys = model.transitions_and_speed_lists_to_numpy_arrays(selected_xs, selected_ys)
            bootstrap_xs, boostrap_ys = model.transitions_and_speed_lists_to_numpy_arrays(bootstrap_xs, boostrap_ys)
            # TRAIN MODEL ON SELECTED CLUSTER
            mlp = model.fit_on_mlp(selected_xs, selected_ys)
            predicts = mlp.predict(bootstrap_xs)
            mse = sklearn.metrics.mean_squared_error(boostrap_ys, predicts)
            mse_sum += mse
            
        errors.append((i, mse_sum/number_of_retries_per_i))

    return errors

def main():
    # fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")
    transitions_speed_dict = encoding.load_transitions_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/files/PlatonSession0BatchAnalysed.csv")
    errors = perform_sampling_test(transitions_speed_dict, sampling_method='uniform')
    
    unzipped = list(zip(*errors))
    xs = unzipped[0]
    ys = unzipped[1]
    plt.plot(xs, ys)
    plt.show()


if __name__ == '__main__':
    main()
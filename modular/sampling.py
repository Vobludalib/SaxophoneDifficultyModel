import sklearn.metrics
import sklearn.model_selection
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
# b) EMPIRICAL (OCCURENCE FREQUENCY) IN WJDB
# c) BASED ON CLUSTERING

# Method for subsampling using uniform randomness
def uniform_subsample(transition_list: list, trill_speeds, n, seed=10, print_debug=False):
    random.seed(seed)

    sampled_trans = []
    sampled_speeds = []

    sampled_indexes = np.array(random.sample([i for i in range(len(transition_list))], n))
    for index in sampled_indexes:
        sampled_trans.append(transition_list[index])
    sampled_speeds = trill_speeds[sampled_indexes]

    return sampled_trans, sampled_speeds

# Method for subsampling using WJDB frequencies
def empirical_subsample(transition_list: list, trill_speeds, n, seed=10, print_debug=False):
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

    occurencesList = [midiToOccurencesDict.get((trans.fingering1.midi, trans.fingering2.midi), 0.1) for trans in transition_list]

    probabilities = np.asarray(occurencesList) / np.sum(occurencesList)
    np.random.seed(seed)
    indexes = np.asarray([i for i in range(len(transition_list))])
    selected_indexes = np.random.choice(a=indexes, p=probabilities, size=n, replace=False)
    selected_trans = []
    selected_speeds = []
    for index in selected_indexes:
        selected_trans.append(transition_list[index])
        selected_speeds.append(trill_speeds[index])

    selected_speeds = np.asarray(selected_speeds)
    return selected_trans, selected_speeds

# Method for subsampling based on clusters
def cluster_subsample(transition_list: list, trill_speeds, n, seed=10, print_debug=False):
    random.seed(seed)

    clusters_dict = {}
    transition_features = [encoding.generate_transition_features(trans, style='expert', expert_weights=False) for trans in transition_list]
    _, labels, _ = skc.k_means(n_clusters=n, X=np.asarray(transition_features))
    for index, transition in enumerate(transition_list):
        label = labels[index]
        if clusters_dict.get(label, None) is None:
            clusters_dict[label] = [(transition, trill_speeds[index])]
        else:
            clusters_dict[label].append((transition, trill_speeds[index]))

    selected_trans = []
    selected_speeds = []
    for cluster_label in clusters_dict:
        sample = random.sample(clusters_dict[cluster_label], 1)[0]
        trans = sample[0]
        speed = sample[1]
        selected_trans.append(trans)
        selected_speeds.append(speed)

    selected_speeds = np.asarray(selected_speeds)
    return selected_trans, selected_speeds

def get_stratified_kfold(xs, ys, test_size, seed=10):
    amount_of_data_points = len(xs)
    amount_of_folds = amount_of_data_points//test_size
    bins = np.array([0, 1.5, 3, 4.5, 10])
    binned_ys = np.digitize(ys, bins)
    skf = sklearn.model_selection.StratifiedKFold(amount_of_folds)
    folds = []
    for i, (train_index, test_index) in enumerate(skf.split(xs, binned_ys)):
        folds.append((i, train_index, test_index))

    return folds

def perform_sampling_test(transitions_trill_speed_dict, sampling_method, size_of_test_set = 20, minimum_amount_of_samples=10):
    # For the sake of the sampling test, for each transition we uniformly randomly select only one of its recorded intervals
    xs = []
    ys = []

    sampling_func = None
    if sampling_method == 'uniform':
        sampling_func = uniform_subsample
    elif sampling_method == 'cluster':
        sampling_func = cluster_subsample
    elif sampling_method == 'empirical':
        sampling_func = empirical_subsample
    else:
        raise NotImplementedError

    for key in transitions_trill_speed_dict:
        if len(transitions_trill_speed_dict[key]) == 1:
            transitions_trill_speed_dict[key] = transitions_trill_speed_dict[key][0]
        else:
            transitions_trill_speed_dict[key] = random.sample(transitions_trill_speed_dict[key], 1)[0]
        
        xs.append(key)
        ys.append(transitions_trill_speed_dict[key])

    ys = np.asarray(ys)

    errors = []

    folds = get_stratified_kfold(xs, ys, test_size=size_of_test_set)
    for i, train_index, test_index in folds:
        errors.append([])
        print(f"=== Doing fold {i} ===")
        train_xs = []
        train_ys = []
        test_xs = []
        test_ys = []
        for train_i in train_index:
            train_xs.append(xs[train_i])
        train_ys = ys[train_index]
        for test_i in test_index:
            test_xs.append(xs[test_i])
        test_ys = ys[test_index]

        # Iterate over every n possible to subsample
        for j in range(train_ys.shape[0] - minimum_amount_of_samples):
            print(f"Sampling only {j + minimum_amount_of_samples} samples out of {train_ys.shape[0]}")
            amount_to_sample = j + minimum_amount_of_samples
            selected_trans, selected_ys = sampling_func(train_xs, train_ys, n=amount_to_sample)
            train_features, train_selected_ys = model.transitions_and_speed_lists_to_numpy_arrays(selected_trans, selected_ys)
            mlp = model.fit_on_lm(train_features, train_selected_ys)
            test_features, test_ys = model.transitions_and_speed_lists_to_numpy_arrays(test_xs, test_ys)
            predicts = mlp.predict(test_features)
            error = sklearn.metrics.mean_squared_error(test_ys, predicts)
            errors[i].append(error)

    return errors

def main():
    # fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")
    transitions_speed_dict = encoding.load_transitions_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/data_with_pilot.csv")
    # TODO: Filter out same-note trills -> huge outliers
    min_samples = 20
    errors = perform_sampling_test(transitions_speed_dict, sampling_method='cluster', minimum_amount_of_samples=min_samples)
    
    for fold_index in range(len(errors)):
        xs = np.array([i + min_samples for i in range(len(errors[fold_index]))])
        ys = np.array(errors[fold_index])
        plt.subplot(3, 3, fold_index + 1)
        plt.plot(xs, ys, label = f"Fold {fold_index}")
        plt.legend()

    plt.show()


if __name__ == '__main__':
    main()
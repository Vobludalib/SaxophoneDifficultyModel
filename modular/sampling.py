import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.neural_network
import encoding
import numpy as np
import sklearn.cluster as skc
import random
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import model
import sklearn
import time
import scipy
import os

# FROM ALL FINGERINGS SUBSAMPLE USING 
# a) UNIFORM RANDOMNESS
# b) EMPIRICAL (OCCURENCE FREQUENCY) IN WJDB
# c) BASED ON CLUSTERING

# Method for subsampling using uniform randomness
def uniform_subsample(transition_list: list, trill_speeds, n):

    sampled_trans = []
    sampled_speeds = []

    sampled_indexes = np.array(random.sample([i for i in range(len(transition_list))], n))
    for index in sampled_indexes:
        sampled_trans.append(transition_list[index])
    sampled_speeds = trill_speeds[sampled_indexes]

    return sampled_trans, sampled_speeds

def parse_bigram_csv_to_dict(path_to_csv, d: None | dict = None):
    if d is None:
        d = {}
        
    def read_row_into_dict(d: dict, string):
        bigrams = str.split(string, ':')
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
    
    with open(path_to_csv, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=";")
        header = next(csvReader, None)
        for row in csvReader:
            read_row_into_dict(d, row[1])

    return d


# Method for subsampling using WJDB frequencies
def empirical_subsample(transition_list: list, trill_speeds, n, cached_dict: None | dict = None):
    if cached_dict is None:
        path_to_csv = "/Users/slibricky/Desktop/Thesis/melospy-gui_V_1_4b_mac_osx/bin/analysis/feature+viz/bigramsTS.csv"

        bigramDict = { }

        bigramDict = parse_bigram_csv_to_dict(path_to_csv, bigramDict)
    else:
        bigramDict = cached_dict

    midiToOccurencesDict = dict(sorted(bigramDict.items(), key= lambda x: x[1]))
    midiToNumberOfTransitionDict = {}
    for trans in transition_list:
        key = (trans[0].midi, trans[1].midi)
        midiToNumberOfTransitionDict[key] = midiToNumberOfTransitionDict.get(key, 0) + 1

    occurencesList = [midiToOccurencesDict.get((trans.fingering1.midi, trans.fingering2.midi), 0.1) / midiToNumberOfTransitionDict.get((trans.fingering1.midi, trans.fingering2.midi), 1) for trans in transition_list]

    probabilities = np.asarray(occurencesList) / np.sum(occurencesList)
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
def cluster_subsample(transition_list: list, trill_speeds, n):
    clusters_dict = {}
    feature_extractor = encoding.ExpertFeatureNumberOfFingersExtractor(use_expert_weights=True, remove_midi=False)
    transition_features = [feature_extractor.get_features(trans) for trans in transition_list]
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

def get_stratified_kfold(xs, ys, test_size):
    amount_of_data_points = len(xs)
    amount_of_folds = amount_of_data_points//test_size
    bins = np.array([0, 1.5, 3, 4.5, 10])
    binned_ys = np.digitize(ys, bins)
    skf = sklearn.model_selection.StratifiedKFold(amount_of_folds, shuffle=True)
    folds = []
    for i, (train_index, test_index) in enumerate(skf.split(xs, binned_ys)):
        folds.append((i, train_index, test_index))

    return folds

def perform_sampling_test(transitions_trill_speed_dict, sampling_method, feature_extractor: encoding.TransitionFeatureExtractor, size_of_test_set = 50, minimum_amount_of_samples=10, amount_of_repeats_per_sampling_point=10, seed=10):
    # For the sake of the sampling test, for each transition we uniformly randomly select only one of its recorded intervals
    xs = []
    ys = []
    random.seed(seed)
    np.random.seed(seed)

    sampling_func = None
    if sampling_method == 'uniform':
        sampling_func = uniform_subsample
    elif sampling_method == 'cluster':
        sampling_func = cluster_subsample
    elif sampling_method == 'empirical':
        sampling_func = empirical_subsample
    else:
        raise NotImplementedError
    
    # Used in empirical sampling and weighted MSE
    cached_dict = parse_bigram_csv_to_dict("/Users/slibricky/Desktop/Thesis/melospy-gui_V_1_4b_mac_osx/bin/analysis/feature+viz/bigramsTS.csv", None)

    for key in transitions_trill_speed_dict:
        if len(transitions_trill_speed_dict[key]) == 1:
            transitions_trill_speed_dict[key] = transitions_trill_speed_dict[key][0]
        else:
            transitions_trill_speed_dict[key] = random.sample(transitions_trill_speed_dict[key], 1)[0]
        
        xs.append(key)
        ys.append(transitions_trill_speed_dict[key])

    ys = np.asarray(ys)

    errors = []
    mapes = []

    folds = get_stratified_kfold(xs, ys, test_size=size_of_test_set)
    for i, train_index, test_index in folds:
        errors.append([])
        mapes.append([])
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
            error_sum = 0
            mape_sum = 0
            kendalls_taus = []
            spearmans = []
            for z in range(amount_of_repeats_per_sampling_point):
                if sampling_func == empirical_subsample:
                    selected_trans, selected_ys = sampling_func(train_xs, train_ys, n=amount_to_sample, cached_dict=cached_dict)
                else:
                    selected_trans, selected_ys = sampling_func(train_xs, train_ys, n=amount_to_sample)
                train_features, train_selected_ys = model.transitions_and_speed_lists_to_numpy_arrays(selected_trans, selected_ys, feature_extractor)
                m = model.TrillSpeedModel(feature_extractor, perform_only_infilling=False)
                m.set_custom_training_data(train_features, train_selected_ys)
                # model_to_use = sklearn.linear_model.LinearRegression()
                model_to_use = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, solver='lbfgs')
                m.train_model(model_to_use)
                test_features, test_ys = model.transitions_and_speed_lists_to_numpy_arrays(test_xs, test_ys, feature_extractor)
                predicts = m.predict(test_features)
                error = sklearn.metrics.mean_squared_error(test_ys, predicts)
                error_sum += error
                mape = sklearn.metrics.mean_absolute_percentage_error(test_ys, predicts)
                mape_sum += mape
                spearman = scipy.stats.spearmanr(test_ys, predicts)
                spearmans.append(spearman.statistic)
                kendalls_tau = scipy.stats.kendalltau(test_ys, predicts)
                kendalls_taus.append(kendalls_tau.statistic)
            error = error_sum / amount_of_repeats_per_sampling_point
            mape = mape_sum / amount_of_repeats_per_sampling_point
            errors[i].append(error)
            mapes[i].append(mape)

    return errors, mapes

def main():
    # fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")
    transitions_speed_dict = encoding.load_transitions_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/files/data_processed/ALL_DATA.csv")
    # Filter out same-note trills -> huge outliers
    to_delete = []
    for key in transitions_speed_dict:
        if key.fingering1.midi == key.fingering2.midi:
            to_delete.append(key)

    for delete in to_delete:
        transitions_speed_dict.pop(delete, None)

    seed = 11
    sampling_method = 'empirical'
    min_samples = 20
    test_set_size = 150 
    amount_of_repeats_per_sampling_point = 3
    feature_extractor = encoding.ExpertFeatureNumberOfFingersExtractor(False, False)
    amount_of_transitions = len(list(transitions_speed_dict.keys()))
    fe = type(feature_extractor).__name__
    if type(feature_extractor) == encoding.ExpertFeatureIndividualFingersExtractor or type(feature_extractor) == encoding.ExpertFeatureNumberOfFingersExtractor:
        fe += f"-{"EW" if feature_extractor.use_expert_weights else "NOEW"}-{"NOMIDI" if feature_extractor.remove_midi else "MIDI"}"
    errors, mapes = perform_sampling_test(transitions_speed_dict, sampling_method=sampling_method, feature_extractor=feature_extractor, size_of_test_set=test_set_size, minimum_amount_of_samples=min_samples, amount_of_repeats_per_sampling_point=amount_of_repeats_per_sampling_point, seed=seed)
    
    random.seed(time.time())
    experiment_id = random.randint(0, 10000000)
    with open(f'./files/sampling_tests/test_{experiment_id}.csv', 'w') as f:
        lines = [f"Min samples: {min_samples}\n", f"Amount of transitions: {amount_of_transitions}\n", f"Test set_size: {test_set_size}\n", f"Sampling method: {sampling_method}\n", f"Amount of repeats per sample point: {amount_of_repeats_per_sampling_point}\n", f"Feature Extractor: {fe}\n", f"Seed: {seed}"]
        f.writelines(lines)
        writer = csv.writer(f)
        writer.writerow(['MSEs'])
        writer.writerows(errors)
        writer.writerow(['MAPEs'])
        writer.writerows(mapes)

    for fold_index in range(len(errors)):
        xs = np.array([i + min_samples for i in range(len(errors[fold_index]))])
        ys = np.array(errors[fold_index])
        ax = plt.gca()
        ax.set_ylim([0, 3])
        plt.plot(xs, ys, label = f"Fold {fold_index}")
        plt.title(f"{sampling_method} sampling, using {fe}")
        plt.xlabel("# of samples")
        plt.ylabel("MSE")
        plt.legend()

    plt.savefig(f"./files/sampling_tests/test_{experiment_id}.png")

    os.system('say "Sampling tests have been completed. At this point I am just speaking to alert to you"')


if __name__ == '__main__':
    main()
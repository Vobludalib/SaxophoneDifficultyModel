# See which approach is best for predicting trill speed given the fingerings
# a) Features (choose only best or what?)
# b) Automatic feature extraction (using what methods?)
# c) Raw encodings into NN

import csv
import matplotlib.patches
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn
import sklearn.neural_network
import sklearn.preprocessing
import encoding
import numpy as np
import random
import sampling
import scipy
import time
import matplotlib.pyplot as plt
import matplotlib
import argparse
import os

def transitions_and_speed_lists_to_numpy_arrays(transitions, speeds, feature_extractor: encoding.TransitionFeatureExtractor):
    features = [feature_extractor.get_features(transition) for transition in transitions]
    xs = np.asarray(features)
    ys = np.asarray(speeds)

    return xs, ys

def transitions_trill_dict_to_numpy_arrays(transitions_to_trill_dict, feature_extractor: encoding.TransitionFeatureExtractor):
    """
    If multiple of the same transition appear in the data, their trill speed is taken as the mean of all the available trill speeds for that transition
    """
    transitions = transitions_to_trill_dict.keys()
    trill_speeds_grouped = transitions_to_trill_dict.values()
    trill_speeds = []
    for trill_speed in trill_speeds_grouped:
        trill_speeds.append(np.sum(trill_speed)/len(trill_speed))

    xs, ys = transitions_and_speed_lists_to_numpy_arrays(transitions, trill_speeds, feature_extractor)

    return xs, ys

class TrillSpeedModel():
    def __init__(self, feature_extractor: encoding.TransitionFeatureExtractor, perform_only_infilling=False, min_trill_speed = 0.5):
        self.feature_extractor = feature_extractor
        self.only_infilling = perform_only_infilling
        self.model = None
        self.xs = None
        self.ys = None
        self.model = None
        self.trained = False
        self.trans_to_trill_dict = {}
        self.min_trill_speed = min_trill_speed

    def load_data_from_csv(self, path):
        """
        Expects a path to a CSV formatted according to the rules set out in encoding.load_transitions_from_file()
        """
        if self.xs is not None or self.ys is not None:
            self.trained = False

        trans_to_trill_dict = encoding.load_transitions_from_file(path)
        self.trans_to_trill_dict = trans_to_trill_dict
        
        self.xs, self.ys = transitions_trill_dict_to_numpy_arrays(trans_to_trill_dict, self.feature_extractor)

    def set_custom_training_data(self, xs, ys):
        if self.xs is not None or self.ys is not None:
            self.trained = False

        if type(xs[0]) == encoding.Transition:
            temp_xs = []
            for trans in xs:
                temp_xs.append(self.feature_extractor.get_features(trans))
            xs = np.asarray(temp_xs)
         
        self.xs, self.ys = xs, ys

    def train_model(self, model):
        self.model = model
        model.fit(self.xs, self.ys)
        self.trained = True

    def predict_transitions(self, transitions: encoding.Transition):
        if not self.trained:
            raise Exception("The model has not been trained since the new data was loaded")
        
        if self.trans_to_trill_dict is not None and self.only_infilling == True:
            results = []
            for transition in transitions:
                if self.trans_to_trill_dict.get(transition, None) is not None:
                    results.append(np.sum(self.trans_to_trill_dict[transition]) / len(self.trans_to_trill_dict[transition]))
                else: 
                    features = np.asarray([self.feature_extractor.get_features(transition)])
                    results.append(self.model.predict(features)[0])
            return np.clip(np.asarray(results), a_min=self.min_trill_speed, a_max=None)
        else:
            features = np.asarray([self.feature_extractor.get_features(transition) for transition in transitions])
            return np.clip(self.model.predict(features), a_min=self.min_trill_speed, a_max=None)
    
    def predict(self, transition_features: np.ndarray):
        if not self.trained:
            raise Exception("The model has not been trained since the new data was loaded")
        
        return np.clip(self.model.predict(transition_features), a_min=self.min_trill_speed, a_max=None)

def predict_fingering_transition(model, fingering1: encoding.Fingering, fingering2: encoding.Fingering):
    if fingering1.midi > fingering2.midi:
        fingering1, fingering2 = fingering2, fingering1
    trans = encoding.Transition(fingering1, fingering2)
    features = model.feature_extractor.get_features(trans).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

def perform_model_test(model_type, feature_extractor, data_csv, output_dir, seed=10):
    transitions_trill_speed_dict = encoding.load_transitions_from_file(data_csv)
    # Filter out same-note trills -> huge outliers
    # These are trills that trill from the one fingering to another of the same note
    # These do not need to be learned, as the fingering estimation model handles same-note transitions as a special case
    to_delete = []
    for key in transitions_trill_speed_dict:
        if key.fingering1.midi == key.fingering2.midi:
            to_delete.append(key)

    for delete in to_delete:
        transitions_trill_speed_dict.pop(delete, None)

    # Load data for weighing MSE
    bigramDict = sampling.parse_bigram_csv_to_dict(os.path.join(".", "files", "bigramsTS.csv"), None)

    midiToOccurencesDict = dict(sorted(bigramDict.items(), key= lambda x: x[1]))
    midiToNumberOfTransitionDict = {}
    for trans in transitions_trill_speed_dict.keys():
        key = (trans[0].midi, trans[1].midi)
        midiToNumberOfTransitionDict[key] = midiToNumberOfTransitionDict.get(key, 0) + 1

    mse_weights = {}
    for trans in transitions_trill_speed_dict.keys():
        key = (trans[0].midi, trans[1].midi)
        mse_weights[key] = midiToOccurencesDict.get((trans.fingering1.midi, trans.fingering2.midi), 0.1) / midiToNumberOfTransitionDict.get((trans.fingering1.midi, trans.fingering2.midi), 1)

    # For the sake of the tests, for each transition we uniformly randomly select only one of its recorded intervals
    # This is to prevent unbalanced train/test sets due to anchor transitions 
    xs = []
    ys = []
    random.seed(seed)
    np.random.seed(seed)

    for key in transitions_trill_speed_dict:
        if len(transitions_trill_speed_dict[key]) == 1:
            transitions_trill_speed_dict[key] = transitions_trill_speed_dict[key][0]
        else:
            transitions_trill_speed_dict[key] = random.sample(transitions_trill_speed_dict[key], 1)[0]
        
        xs.append(key)
        ys.append(transitions_trill_speed_dict[key])

    ys = np.asarray(ys)

    mses = []
    predicts_vs_true = []
    weighted_mses = []
    mapes = []
    spearmans = []
    kendalls_taus = []
    size_of_test_set = 150

    fe = type(feature_extractor).__name__
    if type(feature_extractor) == encoding.ExpertFeatureIndividualFingersExtractor or type(feature_extractor) == encoding.ExpertFeatureNumberOfFingersExtractor:
        fe += f"-{"EW" if feature_extractor.use_expert_weights else "NOEW"}-{"NOMIDI" if feature_extractor.remove_midi else "MIDI"}"
    if type(feature_extractor) == encoding.FingerFeatureExtractor:
        fe += f"-{"Palm-as-finger" if not feature_extractor.map_palm_to_fingers else "Palm-keys-tied-to-finger"}"

    print(f"DOING TEST ON {model_type} with fe {fe}")

    folds = sampling.get_stratified_kfold(xs, ys, test_size=size_of_test_set)
    for i, train_index, test_index in folds:
        print(f"=== Doing fold {i} ===")
        train_xs = []
        train_ys = []
        test_xs = []
        test_ys = []
        test_mse_weights = []
        for train_i in train_index:
            train_xs.append(xs[train_i])
        train_ys = ys[train_index]
        for test_i in test_index:
            test_xs.append(xs[test_i])
            key = (xs[test_i][0].midi, xs[test_i][1].midi)
            test_mse_weights.append(mse_weights[key])
        test_mse_weights = np.asarray(test_mse_weights)
        test_ys = ys[test_index]

        if "log" in model_type:
            func = np.vectorize(lambda x: np.log(x + 1))
            train_ys = func(train_ys)

        train_features, train_selected_ys = transitions_and_speed_lists_to_numpy_arrays(train_xs, train_ys, feature_extractor)
        m = TrillSpeedModel(feature_extractor, perform_only_infilling=False)
        m.set_custom_training_data(train_features, train_selected_ys)

        model_to_use = None
        if "lm" in model_type:
            model_to_use = sklearn.linear_model.LinearRegression()
        elif "mlp" in model_type:
            model_to_use = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, solver='lbfgs')
        else:
            raise NotImplementedError()
        
        m.train_model(model_to_use)
        test_features, test_ys = transitions_and_speed_lists_to_numpy_arrays(test_xs, test_ys, feature_extractor)
        predicts = m.predict(test_features)
        if "log" in model_type:
            exp = np.vectorize(lambda x: np.exp(x) - 1)
            predicts = exp(predicts)
        predicts_vs_true.append(np.vstack([predicts, test_ys]))
        mse = sklearn.metrics.mean_squared_error(test_ys, predicts)
        mses.append(mse)
        weighted_mse = sklearn.metrics.mean_squared_error(test_ys, predicts, sample_weight=test_mse_weights)
        weighted_mses.append(weighted_mse)
        mape = sklearn.metrics.mean_absolute_percentage_error(test_ys, predicts)
        mapes.append(mape)
        spearman = scipy.stats.spearmanr(test_ys, predicts)
        spearmans.append(spearman.statistic)
        kendalls_tau = scipy.stats.kendalltau(test_ys, predicts)
        kendalls_taus.append(kendalls_tau.statistic)

    random.seed(time.time())
    experiment_id = random.randint(0, 10000000)
    with open(os.path.join(output_dir, f"{model_type}_{fe}_{experiment_id}.csv"), "w") as f:
        f.writelines([f"Model type: {model_type}\n", f"Size of test set: {size_of_test_set}\n", f"Number of folds: {i + 1}\n", f"Feature Extractor: {fe}\n", f"Seed {seed}\n"])
        writer = csv.writer(f, lineterminator='\n')
        f.write("MSES:\n")
        writer.writerow(mses)
        f.write("Weighted MSES:\n")
        writer.writerow(weighted_mses)
        f.write("Mean absolute percent error:\n")
        writer.writerow(mapes)
        f.write("Spearman's:\n")
        writer.writerow(spearmans)
        f.write("Kendall's taus:\n")
        writer.writerow(kendalls_taus)
        f.write(f"Average MSE over folds: {np.mean(mses):.2f}\n")
        f.write(f"Average wMSE over folds: {np.mean(weighted_mses):.2f}\n")
        f.write(f"Average MAPE over folds: {np.mean(mapes):.2f}\n")
        f.write(f"Average Spearman's over folds: {np.mean(mses):.2f}\n")
        f.write(f"Average Kendall's Tau over folds: {np.mean(kendalls_taus):.2f}\n")
        f.write(f"Latex table format: {np.mean(mses):.2f} & {np.mean(weighted_mses):.2f} & {np.mean(mapes):.2f}")

    i = 0
    graph_xs = []
    ys = []
    colors = []
    for fold_i, pred_vs_true in enumerate(predicts_vs_true):
        if fold_i > 0:
            break
        graph_xs += [int(i + j/2) for j in range(pred_vs_true.shape[1]*2)]
        sorted_indices = np.argsort(pred_vs_true[1])
        pred_vs_true = pred_vs_true[:, sorted_indices]
        preds = pred_vs_true[0]
        true = pred_vs_true[1]
        for z in range(preds.shape[0]):
            ys.append(true[z])
            colors.append("#648FFF")
            ys.append(preds[z])
            colors.append("#FE6100")
            index = int(i + z)
            plt.annotate('', xy=(index, preds[z]), xycoords='data', xytext=(index, true[z]), textcoords='data', arrowprops=dict(facecolor='black', arrowstyle='->'))
        i += pred_vs_true.shape[1]

    plt.scatter(graph_xs, ys, color=colors)
    plt.xlabel("Individual transitions from fold 0 test data sorted by true trill speed")
    plt.ylabel("Trill speeds in trills/s")
    legend_handles = [
        matplotlib.patches.Patch(facecolor="#648FFF", label="True (recorded) trill speed"),
        matplotlib.patches.Patch(facecolor="#FE6100", label="Predicted trill speed")
                    ]
    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    ax.legend(handles=legend_handles, loc="upper left")
    plt.title(f"True vs predicted for {model_type} with {fe}")
    plt.savefig(os.path.join(output_dir, f"{model_type}_{fe}_{experiment_id}.png"))
    plt.clf()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, required=True, help="Path to directory where output files will be stored")
    parser.add_argument('-d', '--data', type=str, required=True, help="Path to a .csv file that contains all the data on which to operate. This should be Processed_Data.csv unless you have generated a different file for this purpose.")
    parser.add_argument('-m', '--model', type=str, required=True, choices=['lm', 'mlp', 'mlp-log', 'lm-log'], help="Sets the specific model to use")
    args = parser.parse_args()

    expert_feature_extractors = []
    for extractor_type in [encoding.ExpertFeatureIndividualFingersExtractor, encoding.ExpertFeatureNumberOfFingersExtractor]:
        for use_expert_weights in [True, False]:
            for remove_midi in [True, False]:
                expert_feature_extractors.append(extractor_type(use_expert_weights, remove_midi))

    fes = [encoding.RawFeatureExtractor(), encoding.FingerFeatureExtractor(map_palm_to_fingers=True), encoding.FingerFeatureExtractor(map_palm_to_fingers=False)] + expert_feature_extractors

    for feature_extractor in fes:
        perform_model_test(args.model, feature_extractor, args.data, args.out)

if __name__ == '__main__':
    main()

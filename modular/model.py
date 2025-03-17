# See which approach is best for predicting trill speed given the fingerings
# a) Features (choose only best or what?)
# b) Automatic feature extraction (using what methods?)
# c) Raw encodings into NN

import csv
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
    def __init__(self, feature_extractor: encoding.TransitionFeatureExtractor, perform_only_infilling=False):
        self.feature_extractor = feature_extractor
        self.only_infilling = perform_only_infilling
        self.model = None
        self.xs = None
        self.ys = None
        self.model = None
        self.trained = False

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
            return np.asarray(results)
        else:
            features = np.asarray([self.feature_extractor.get_features(transition) for transition in transitions])
            return self.model.predict(features)
    
    def predict(self, transition_features: np.ndarray):
        if not self.trained:
            raise Exception("The model has not been trained since the new data was loaded")
        
        return self.model.predict(transition_features)

def predict_fingering_transition(model, fingering1: encoding.Fingering, fingering2: encoding.Fingering, feature_extractor: encoding.TransitionFeatureExtractor):
    if fingering1.midi > fingering2.midi:
        fingering1, fingering2 = fingering2, fingering1
    trans = encoding.Transition(fingering1, fingering2)
    features = feature_extractor.get_features(trans).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

def main(model_type, feature_extractor):
    transitions_trill_speed_dict = encoding.load_transitions_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/files/normalisation_csvs/ALL_DATA.csv")
    # Filter out same-note trills -> huge outliers
    to_delete = []
    for key in transitions_trill_speed_dict:
        if key.fingering1.midi == key.fingering2.midi:
            to_delete.append(key)

    for delete in to_delete:
        transitions_trill_speed_dict.pop(delete, None)

    seed = 10

    # For the sake of the sampling test, for each transition we uniformly randomly select only one of its recorded intervals
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

    errors = []
    spearmans = []
    kendalls_taus = []
    size_of_test_set = 150
    # model_type = "lm"
    # model_type = "mlp"

    # feature_extractor = encoding.ExpertFeatureIndividualFingersExtractor()
    fe = type(feature_extractor).__name__
    if type(feature_extractor) == encoding.ExpertFeatureIndividualFingersExtractor or type(feature_extractor) == encoding.ExpertFeatureNumberOfFingersExtractor:
        fe += f"-{"EW" if feature_extractor.use_expert_weights else "NOEW"}-{"NOMIDI" if feature_extractor.remove_midi else "MIDI"}"

    print(f"DOING TEST ON {model_type} with fe {fe}")

    folds = sampling.get_stratified_kfold(xs, ys, test_size=size_of_test_set)
    for i, train_index, test_index in folds:
        print(test_index)
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

        train_features, train_selected_ys = transitions_and_speed_lists_to_numpy_arrays(train_xs, train_ys, feature_extractor)
        m = TrillSpeedModel(feature_extractor, perform_only_infilling=False)
        m.set_custom_training_data(train_features, train_selected_ys)

        model_to_use = None
        if model_type == "lm":
            model_to_use = sklearn.linear_model.LinearRegression()
        elif model_type == "mlp":
            model_to_use = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,), max_iter=3000)
        else:
            raise NotImplementedError()
        
        m.train_model(model_to_use)
        test_features, test_ys = transitions_and_speed_lists_to_numpy_arrays(test_xs, test_ys, feature_extractor)
        predicts = m.predict(test_features)
        error = sklearn.metrics.mean_squared_error(test_ys, predicts)
        errors.append(error)
        spearman = scipy.stats.spearmanr(test_ys, predicts)
        spearmans.append(spearman.statistic)
        kendalls_tau = scipy.stats.kendalltau(test_ys, predicts)
        kendalls_taus.append(kendalls_tau.statistic)

    random.seed(time.time())
    experiment_id = random.randint(0, 10000000)
    with open(f"./files/model_tests/{model_type}_{fe}_{experiment_id}.csv", "w") as f:
        f.writelines([f"Model type: {model_type}\n", f"Size of test set: {size_of_test_set}\n", f"Number of folds: {i + 1}\n", f"Feature Extractor: {fe}\n", f"Seed {seed}\n"])
        writer = csv.writer(f)
        writer.writerow(errors)
        writer.writerow(spearmans)
        writer.writerow(kendalls_taus)
        f.write(f"Average MSE over folds: {np.mean(errors)}")

if __name__ == '__main__':
    expert_feature_extractors = []
    for extractor_type in [encoding.ExpertFeatureIndividualFingersExtractor, encoding.ExpertFeatureNumberOfFingersExtractor]:
        for use_expert_weights in [True, False]:
            for remove_midi in [True, False]:
                expert_feature_extractors.append(extractor_type(use_expert_weights, remove_midi))

    fes = [encoding.RawFeatureExtractor(), encoding.FingerFeatureExtractor()] + expert_feature_extractors

    for model_type in ["mlp", "lm"]:
        for feature_extractor in fes:
            main(model_type, feature_extractor)

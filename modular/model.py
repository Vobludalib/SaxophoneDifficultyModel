# See which approach is best for predicting trill speed given the fingerings
# a) Features (choose only best or what?)
# b) Automatic feature extraction (using what methods?)
# c) Raw encodings into NN

import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn
import sklearn.neural_network
import sklearn.preprocessing
import encoding
import numpy as np

def transitions_and_speed_lists_to_numpy_arrays(transitions, speeds, feature_extractor: encoding.RawFeatureExtractor | encoding.ExpertFeatureExtractor):
    features = [feature_extractor.get_features(transition) for transition in transitions]
    xs = np.asarray(features)
    ys = np.asarray(speeds)

    return xs, ys

def transitions_trill_dict_to_numpy_arrays(transitions_to_trill_dict, feature_extractor: encoding.RawFeatureExtractor | encoding.ExpertFeatureExtractor):
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

class FingeringTransitionModel():
    def __init__(self, feature_extractor: encoding.RawFeatureExtractor | encoding.ExpertFeatureExtractor, perform_only_infilling=False):
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
            return np.asarray(results)
        else:
            features = np.asarray([self.feature_extractor.get_features(transition) for transition in transitions])
            return self.model.predict(features)
    
    def predict(self, transition_features: np.ndarray):
        if not self.trained:
            raise Exception("The model has not been trained since the new data was loaded")
        
        return self.model.predict(transition_features)

def fit_on_mlp(xs, ys) -> sklearn.neural_network.MLPRegressor:
    mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(20,), max_iter=3000)
    mlp.fit(xs, ys)
    return mlp

def fit_on_lm(xs, ys) -> sklearn.linear_model.LinearRegression:
    lm = sklearn.linear_model.LinearRegression()
    lm.fit(xs, ys)
    return lm

def predict_fingering_transition(model, fingering1: encoding.Fingering, fingering2: encoding.Fingering, feature_extractor: encoding.RawFeatureExtractor | encoding.ExpertFeatureExtractor):
    if fingering1.midi > fingering2.midi:
        fingering1, fingering2 = fingering2, fingering1
    trans = encoding.Transition(fingering1, fingering2)
    features = feature_extractor.get_features(trans).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

def main():
    transitions_to_trill_dict = encoding.load_transitions_from_file('/Users/slibricky/Desktop/Thesis/thesis/modular/files/PlatonSession0BatchAnalysed.csv')
    xs, ys = transitions_trill_dict_to_numpy_arrays(transitions_to_trill_dict)
    xs = xs[:,2:]

    train_xs, test_xs, train_ys, test_ys = sklearn.model_selection.train_test_split(xs, ys, test_size=0.1)

    mlp = fit_on_mlp(train_xs, train_ys)
    predicts = mlp.predict(test_xs)
    mse = sklearn.metrics.mean_squared_error(test_ys, predicts)
    print(mse)
    comparison = np.vstack([test_ys, predicts])
    print(comparison.T)


if __name__ == '__main__':
    main()

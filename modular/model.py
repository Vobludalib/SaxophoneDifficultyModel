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

def fit_on_mlp(xs, ys) -> sklearn.neural_network.MLPRegressor:
    mlp = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(20,), max_iter=3000)
    mlp.fit(xs, ys)
    return mlp

def fit_on_lm(xs, ys) -> sklearn.linear_model.LinearRegression:
    lm = sklearn.linear_model.LinearRegression()
    lm.fit(xs, ys)
    return lm

def transitions_trill_dict_to_numpy_arrays(transitions_to_trill_dict):
    transitions = transitions_to_trill_dict.keys()
    trill_speeds_grouped = transitions_to_trill_dict.values()
    trill_speeds = []
    for trill_speed in trill_speeds_grouped:
        trill_speeds.append(np.sum(trill_speed)/len(trill_speed))

    xs, ys = transitions_and_speed_lists_to_numpy_arrays(transitions, trill_speeds)

    return xs, ys

def transitions_and_speed_lists_to_numpy_arrays(transitions, speeds):
    # For now using this features method -> possibly to be changed for some experiments
    features = [encoding.generate_transition_features(trans, style='expert', expert_weights=True, remove_midi=False) for trans in transitions]
    xs = np.asarray(features)
    ys = np.asarray(speeds)

    return xs, ys

def predict_fingering_transition(model, fingering1: encoding.Fingering, fingering2: encoding.Fingering):
    features = encoding.generate_transition_features(encoding.Transition(fingering1, fingering2)).reshape(1, -1)
    prediction = model.predict(features)
    return prediction

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

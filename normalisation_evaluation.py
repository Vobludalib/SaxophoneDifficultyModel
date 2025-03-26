import sklearn.metrics
import encoding
import normalisation_tool
import model
import sampling
import os
import numpy as np
import sklearn

def load_transitions_from_directory(directory_path):
    file_to_anchors_dict = {}
    for file in os.listdir(directory_path):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            full_path = os.path.join(directory_path, filename)
            file_to_anchors_dict[filename] = encoding.load_transitions_from_file(full_path)

    return file_to_anchors_dict

def main():
    # Load data (and anchors) from sessions
    session_to_transitions = load_transitions_from_directory("/Users/slibricky/Desktop/Thesis/thesis/modular/files/data_processed/")
    file_to_anchors_dict = normalisation_tool.load_anchors_from_directory("/Users/slibricky/Desktop/Thesis/thesis/modular/files/data_processed/")
    norm_strength = 0.2

    xs = []
    ys = []
    session_names = []
    for i, session in enumerate(session_to_transitions):
        for transition in session_to_transitions[session]:
            for speed in session_to_transitions[session][transition]:
                xs.append(transition)
                ys.append(speed)
                session_names.append(session)

    # Precalculate normalisation values
    anchor_transitions_sorted, speeds = normalisation_tool.calculate_anchor_speeds(file_to_anchors_dict)
    averages = np.mean(speeds, axis=0)
    differences = normalisation_tool.calculate_difference_to_mean(speeds, averages)
    feature_extractor = encoding.ExpertFeatureNumberOfFingersExtractor()
    anchor_features = normalisation_tool.get_anchor_transition_features(anchor_transitions_sorted, feature_extractor)

    session_indexes_for_normalisation = [list(file_to_anchors_dict.keys()).index(session_name) for session_name in session_names]

    # Make a normalised copy of the data
    ys_normalised = []
    for i, y in enumerate(ys):
        ys_normalised.append(normalisation_tool.normalise_transition(xs[i], y, session_indexes_for_normalisation[i], anchor_features, differences, feature_extractor, norm_strength))

    # Split data using stratified k-fold from sampling.py
    folds = sampling.get_stratified_kfold(xs, ys, test_size=30)

    ys = np.asarray(ys)
    ys_normalised = np.asarray(ys_normalised)
    # For each fold:
    for fold_index, train_indexes, test_indexes in folds:
        # Create the normalised and unnormalised training and test data
        train_xs = []
        train_ys = []
        train_ys_norm = []
        test_xs = []
        test_ys = []
        test_ys_norm = []
        test_ys_session_indexes = []
        for train_i in train_indexes:
            train_xs.append(xs[train_i])
        train_ys = ys[train_indexes]
        train_ys_norm = ys_normalised[train_indexes]
        test_ys_norm = ys_normalised[test_indexes]
        for test_i in test_indexes:
            test_xs.append(xs[test_i])
            test_ys_session_indexes.append(session_indexes_for_normalisation[test_i])
        test_ys = ys[test_indexes]

        # Train a model on the unnormalised, and normalised data respectively
        train_xs, train_ys = model.transitions_and_speed_lists_to_numpy_arrays(train_xs, train_ys, feature_extractor)
        no_norm_model = model.fit_on_lm(train_xs, train_ys)
        norm_model = model.fit_on_lm(train_xs, train_ys_norm)

        # Predict the test set using both models
        test_xs, test_ys = model.transitions_and_speed_lists_to_numpy_arrays(test_xs, test_ys, feature_extractor)
        no_norm_predicts = no_norm_model.predict(test_xs)
        norm_predicts = norm_model.predict(test_xs)

        # For the normalised model, perform inverse normalisation
        denormalised_predicts = [normalisation_tool.inverse_normalise_transition(test_xs[i], norm_predicts[i], test_ys_session_indexes[i], anchor_features, differences, norm_strength) for i in range(norm_predicts.shape[0])]

        # Evaluate MSE for both (given inverse normalisation occured, these MSEs are comparable)
        no_norm_mse = sklearn.metrics.mean_squared_error(test_ys, no_norm_predicts)
        norm_mse = sklearn.metrics.mean_squared_error(test_ys, denormalised_predicts)
        no_inverse_denorm_mse = sklearn.metrics.mean_squared_error(test_ys_norm, norm_predicts)
        print(f"Fold {fold_index}")
        print(f"No normalisation MSE: {no_norm_mse}")
        print(f"Inverse normalisation MSE: {norm_mse}")
        print(f"Normalised MSE (without inverse): {no_inverse_denorm_mse}")

if __name__ == '__main__':
    main()
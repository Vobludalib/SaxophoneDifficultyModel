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
    session_to_transitions = load_transitions_from_directory("/Users/slibricky/Desktop/Thesis/thesis/files/data_processed/")
    normaliser = normalisation_tool.MultiplicativeAnchorBasedNormaliser("/Users/slibricky/Desktop/Thesis/thesis/files/data_processed/", norm_strength=0.2, feature_extractor=encoding.ExpertFeatureNumberOfFingersExtractor())

    session_order = list(normaliser.file_to_anchors_dict.keys())

    xs = []
    ys = []
    session_names = []
    for i, session in enumerate(session_to_transitions):
        for transition in session_to_transitions[session]:
            for speed in session_to_transitions[session][transition]:
                if int(speed) == 0:
                    continue
                xs.append(transition)
                ys.append(speed)
                session_names.append(session)

    session_indexes_for_normalisation = [session_order.index(session_name) for session_name in session_names]

    # Make a normalised copy of the data
    ys_normalised = []
    for i, y in enumerate(ys):
        ys_normalised.append(normaliser.normalise(xs[i], y, session_index=session_order.index(session_names[i])))

    # Split data using stratified k-fold from sampling.py
    folds = sampling.get_stratified_kfold(xs, ys, test_size=30)

    ys = np.asarray(ys)
    ys_normalised = np.asarray(ys_normalised)
    no_norm_mses = []
    no_norm_mapes = []
    inv_norm_mses = []
    inv_norm_mapes = []
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
        mNoNorm = model.TrillSpeedModel(encoding.ExpertFeatureNumberOfFingersExtractor())
        mNoNorm.set_custom_training_data(train_xs, train_ys)
        mNoNorm.train_model(sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, solver='lbfgs'))
        mNorm = model.TrillSpeedModel(encoding.ExpertFeatureNumberOfFingersExtractor())
        mNorm.set_custom_training_data(train_xs, train_ys_norm)
        mNorm.train_model(sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, solver='lbfgs'))

        # Predict the test set using both models
        no_norm_predicts = mNoNorm.predict_transitions(test_xs)
        norm_predicts = mNorm.predict_transitions(test_xs)

        # For the normalised model, perform inverse normalisation
        denormalised_predicts = [normaliser.inverse_normalise(test_xs[i], norm_predicts[i], test_ys_session_indexes[i]) for i in range(norm_predicts.shape[0])]

        # Evaluate MSE for both (given inverse normalisation occured, these MSEs are comparable)
        no_norm_mse = sklearn.metrics.mean_squared_error(test_ys, no_norm_predicts)
        no_norm_mape = sklearn.metrics.mean_absolute_percentage_error(test_ys, no_norm_predicts)
        inv_norm_mse = sklearn.metrics.mean_squared_error(test_ys, denormalised_predicts)
        inv_norm_mape = sklearn.metrics.mean_absolute_percentage_error(test_ys, denormalised_predicts)
        no_norm_mses.append(no_norm_mse)
        no_norm_mapes.append(no_norm_mape)
        inv_norm_mses.append(inv_norm_mse)
        inv_norm_mapes.append(inv_norm_mape)
        print(f"Fold {fold_index}")
        print(f"No normalisation MSE: {no_norm_mse}")
        print(f"Inverse normalisation MSE: {inv_norm_mse}")
        print(f"No normalisation MAPE: {no_norm_mape}")
        print(f"Inverse normalisation MAPE: {inv_norm_mape}")

    print(f"No norm MSE average: {np.mean(no_norm_mses)}")
    print(f"Inv norm MSE average: {np.mean(inv_norm_mses)}")
    print(f"No norm MAPE average: {np.mean(no_norm_mapes)}")
    print(f"Inv norm MAPE average: {np.mean(inv_norm_mapes)}")

if __name__ == '__main__':
    main()
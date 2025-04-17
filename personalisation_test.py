import sklearn.metrics
from normalisation_tool import *
import encoding
from encoding import Fingering
from model import TrillSpeedModel
import sklearn

def main():
    parser = argparse.ArgumentParser(
                    prog='Session Normalisation Tool',
                    description='Tool designed to normalise trill speeds between sessions and players from batch-analysed csv files')
    parser.add_argument('--csvs', type=str, help="Path to the folder of batch-analysed .csv files")
    parser.add_argument('-o', type=str, help="Will store all the outputs into one csv file with this name")
    parser.add_argument("--strength", type=float, help="Specify the normalisation strength")
    args = parser.parse_args()
    if (not os.path.isdir(args.csvs)):
        print("--csvs flag must be set to a directory containing .csv files!")
        input()
        exit()

    # Read all the anchors from every csv into dict {filename: [(encoding1, encoding2, speed)]}
    base_normaliser = MultiplicativeAnchorBasedNormaliser(args.csvs, norm_strength=0.2, feature_extractor=encoding.ExpertFeatureNumberOfFingersExtractor())

    for test_i, test_filename in enumerate(base_normaliser.file_to_anchors_dict):
        print(f"=== Performing test {test_i} eliminating file: {test_filename} ===")
        test_normaliser = MultiplicativeAnchorBasedNormaliser(args.csvs, norm_strength=0.2, feature_extractor=encoding.ExpertFeatureNumberOfFingersExtractor())
        # Precompute anchor information for the test anchors
        test_anchor_speeds = test_normaliser.speeds[test_i]
        del test_normaliser.file_to_anchors_dict[test_filename]
        test_normaliser.precalculate_values()
        # Then recalculate the values without the test session included

        train_xs = []
        train_ys = []
        for i, filename in enumerate(test_normaliser.file_to_anchors_dict):
            full_path = os.path.join(args.csvs, filename)
            with open(full_path, 'r') as csvf:
                reader = csv.reader(csvf)
                next(reader, None)
                for row in reader:
                    filename = row[0]
                    cluster = int(row[1])
                    midi1 = int(row[2])
                    name1 = row[3]
                    encoding1 = row[4]
                    midi2 = int(row[5])
                    name2 = row[6]
                    encoding2 = row[7]
                    speed = float(row[8])
                    fingering1 = Fingering(midi1, name1, encoding1)
                    fingering2 = Fingering(midi2, name2, encoding2)
                    transition = encoding.Transition(fingering1, fingering2)
                    normalised_speed = test_normaliser.normalise(transition, speed, test_anchor_speeds)
                    train_xs.append(test_normaliser.feature_extractor.get_features(transition))
                    train_ys.append(normalised_speed)

        tsm = TrillSpeedModel(test_normaliser.feature_extractor)
        tsm.set_custom_training_data(train_xs, train_ys)
        tsm.train_model(sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, solver='lbfgs'))

        full_path = os.path.join(args.csvs, test_filename)
        predicted_non_norm_speeds = []
        predicted_norm_speeds = []
        actual_non_norm_speeds = []
        with open(full_path, 'r') as csvf:
            reader = csv.reader(csvf)
            next(reader, None)
            for row in reader:
                cluster = int(row[1])
                if cluster == -1:
                    # Test isn't performed on anchor transitions, as it makes little sense to do so
                    continue
                midi1 = int(row[2])
                name1 = row[3]
                encoding1 = row[4]
                midi2 = int(row[5])
                name2 = row[6]
                encoding2 = row[7]
                non_normalised_speed = float(row[8])
                if int(non_normalised_speed) == 0:
                    # Ignore same note transitions
                    continue
                actual_non_norm_speeds.append(non_normalised_speed)
                fingering1 = Fingering(midi1, name1, encoding1)
                fingering2 = Fingering(midi2, name2, encoding2)
                transition = encoding.Transition(fingering1, fingering2)
                features = test_normaliser.feature_extractor.get_features(transition)
                predicted_normalised_speed =  tsm.predict(features.reshape((1, -1)))
                predicted_norm_speeds.append(predicted_normalised_speed)
                predicted_non_normalised_speed = test_normaliser.inverse_normalise(transition, predicted_normalised_speed, test_anchor_speeds)
                predicted_non_norm_speeds.append(predicted_non_normalised_speed)
                print(f"Actual ts: {non_normalised_speed}")
                print(f"TS predicted before invnorm: {predicted_normalised_speed}")
                print(f"TS after invnorm: {predicted_non_normalised_speed}")

        mse_non_norms = sklearn.metrics.mean_squared_error(actual_non_norm_speeds, predicted_non_norm_speeds)
        mse_norm_to_non = sklearn.metrics.mean_squared_error(actual_non_norm_speeds, predicted_norm_speeds)
        mape_non_norms = sklearn.metrics.mean_absolute_percentage_error(actual_non_norm_speeds, predicted_non_norm_speeds)
        mape_norm_to_non = sklearn.metrics.mean_absolute_percentage_error(actual_non_norm_speeds, predicted_norm_speeds)
        print(f"MSE non to non: {mse_non_norms}")
        print(f"MSE norm to non: {mse_norm_to_non}")
        print(f"MAPE non to non: {mape_non_norms}")
        print(f"MAPE norm to non: {mape_norm_to_non}")
        input()

if __name__ == '__main__':
    main()
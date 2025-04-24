import csv
from encoding import Fingering
import encoding
import argparse
import os
import numpy as np
import scipy.special
import abc
import scipy.stats.mstats

# Takes seperate batch_analysed csvs (each corresponding to one session) and generates one big CSV of trills
# normalised using the anchor transitions

# ANCHORS ARE TAKEN AS BEING CLUSTER -1

def get_euclidean_distance(features1, features2):
        return np.sum([((features1[i] - features2[i]) ** 2) ** 0.5 for i in range(features1.shape[0])])

class NormalisationTool(abc.ABC):
    pass

class AnchorBasedNormaliser(NormalisationTool):
    @staticmethod
    def load_anchors_from_file(file_path) -> list:
        output = []
        with open(file_path, 'r') as csvf:
            reader = csv.reader(csvf)
            next(reader, None)
            for row in reader:
                cluster = int(row[1])
                if (cluster == -1):
                    midi1 = int(row[2])
                    name1 = row[3]
                    encoding1 = row[4]
                    midi2 = int(row[5])
                    name2 = row[6]
                    encoding2 = row[7]
                    speed = float(row[8])
                    output.append((Fingering(midi1, name1, encoding1), Fingering(midi2, name2, encoding2), speed))

        return output

    @staticmethod
    def load_anchors_from_directory(directory_path) -> dict:
        file_to_anchors_dict = {}
        for file in os.listdir(directory_path):
            filename = os.fsdecode(file)
            if filename.endswith('.csv'):
                full_path = os.path.join(directory_path, filename)
                file_to_anchors_dict[filename] = AnchorBasedNormaliser.load_anchors_from_file(full_path)

        return file_to_anchors_dict

    @staticmethod
    def calculate_anchor_speeds(file_to_anchors_dict) -> tuple[list, list]:
        """
        Return order is: list of transition objects, speeds.
        Speeds is a 2d list, row = session, column = anchor transition
        """

        anchor_transitions_sorted = []
        checked = False
        for filename in file_to_anchors_dict:
            file_to_anchors_dict[filename].sort(key = lambda tup: (tup[0].midi, tup[1].midi))
            if not checked:
                for tup in file_to_anchors_dict[filename]:
                    anchor_transitions_sorted.append((tup[0], tup[1]))
                checked = True

        speeds = np.zeros((len(file_to_anchors_dict), len(anchor_transitions_sorted)), dtype=float)
        for row, filename in enumerate(file_to_anchors_dict):
            for column, transition_tup in enumerate(file_to_anchors_dict[filename]):
                speeds[row, column] = transition_tup[2]

        return anchor_transitions_sorted, speeds

    @staticmethod
    def get_anchor_transition_features(anchor_transitions, feature_extractor: encoding.TransitionFeatureExtractor):
        anchor_features = []
        for transition in anchor_transitions:
            anchor_features.append(feature_extractor.get_features(transition))
        anchor_features = np.asarray(anchor_features)
        return anchor_features

class AdditiveAnchorBasedNormaliser(AnchorBasedNormaliser):
    def __init__(self, data_path, norm_strength, feature_extractor: encoding.TransitionFeatureExtractor):
        self.norm_strength = norm_strength
        self.feature_extractor = feature_extractor
        self.file_to_anchors_dict = self.load_anchors_from_directory(data_path)

        self.precalculate_values()

    def precalculate_values(self):
        self.anchor_transitions_sorted, self.speeds = self.calculate_anchor_speeds(self.file_to_anchors_dict)
        self.averages = np.mean(self.speeds, axis=0)

        self.differences = self.calculate_difference_to_mean(self.speeds, self.averages)

        self.anchor_features = self.get_anchor_transition_features(self.anchor_transitions_sorted, self.feature_extractor)

    @staticmethod
    def calculate_difference_to_mean(speeds, means):
        if speeds.ndim == 1:
            differences = np.zeros((speeds.shape[0],), dtype=float)
            for value in range(means.shape[0]):
                    differences[value] = means[value] - speeds[value]
        else:
            differences = np.zeros((speeds.shape[0], speeds.shape[1]), dtype=float)
            for row in range(speeds.shape[0]):
                for column in range(speeds.shape[1]):
                    differences[row, column] = means[column] - speeds[row, column]

        return differences

    def get_normalisation_adjustment(self, transition, differences):
        transition_features = self.feature_extractor.get_features(transition)
        distances_to_anchors = np.asarray([get_euclidean_distance(transition_features, self.anchor_features[j]) for j in range(self.anchor_features.shape[0])])
        multiples = np.asarray(scipy.special.softmax(distances_to_anchors))
        overall_adjustment = np.sum(np.multiply(multiples, differences)) * self.norm_strength
        return overall_adjustment

    def normalise_by_session_index(self, transition, trill_speed, session_index: int):
        normalised_speed = trill_speed + self.get_normalisation_adjustment(transition, self.differences[session_index])
        return normalised_speed

    def normalise(self, transition, trill_speed, anchor_speeds):
        differences = self.calculate_difference_to_mean(anchor_speeds, self.averages)
        normalised_speed = trill_speed + self.get_normalisation_adjustment(transition, differences)
        return normalised_speed

    def inverse_normalise_by_session_index(self, transition, normalised_trill_speed, session_index: int):
        return normalised_trill_speed - self.get_normalisation_adjustment(transition, self.differences[session_index])
    
    def inverse_normalise(self, transition, normalised_trill_speed, anchor_speeds):
        differences = self.calculate_difference_to_mean(anchor_speeds, self.averages)
        return normalised_trill_speed - self.get_normalisation_adjustment(transition, differences)
        
class MultiplicativeAnchorBasedNormaliser(AnchorBasedNormaliser):
    def __init__(self, data_path, norm_strength, feature_extractor: encoding.TransitionFeatureExtractor):
        self.norm_strength = norm_strength
        self.feature_extractor = feature_extractor
        self.file_to_anchors_dict = self.load_anchors_from_directory(data_path)

        self.precalculate_values()

    def precalculate_values(self):
        self.anchor_transitions_sorted, self.speeds = self.calculate_anchor_speeds(self.file_to_anchors_dict)
        self.means = np.mean(self.speeds, axis=0)

        self.ratios = self.calculate_ratios_with_means(self.speeds, self.means)

        self.anchor_features = self.get_anchor_transition_features(self.anchor_transitions_sorted, self.feature_extractor)

    @staticmethod
    def calculate_ratios_with_means(speeds, means):
        if speeds.ndim == 1:
            ratios = np.zeros((speeds.shape[0],), dtype=float)
            for value in range(ratios.shape[0]):
                    ratios[value] = means[value] / speeds[value]
        else:
            ratios = np.zeros((speeds.shape[0], speeds.shape[1]), dtype=float)
            for row in range(ratios.shape[0]):
                for column in range(ratios.shape[1]):
                    ratios[row, column] = means[column] / speeds[row, column]

        return ratios

    def get_normalisation_coefficient(self, transition, ratios):
        transition_features = self.feature_extractor.get_features(transition)
        distances_to_anchors = np.asarray([get_euclidean_distance(transition_features, self.anchor_features[j]) for j in range(self.anchor_features.shape[0])])
        multiples = np.asarray(scipy.special.softmax(distances_to_anchors))
        overall_coefficient = scipy.stats.mstats.gmean(ratios, axis=0, weights=multiples)
        return overall_coefficient

    def normalise_by_session_index(self, transition, trill_speed, session_index: int):
        normalised_speed = (1-self.norm_strength) * trill_speed + self.norm_strength * (trill_speed * self.get_normalisation_coefficient(transition, self.ratios[session_index]))
        return normalised_speed
    
    def normalise(self, transition, trill_speed, anchor_speeds):
        ratios = self.calculate_ratios_with_means(anchor_speeds, self.means)
        normalised_speed = (1-self.norm_strength) * trill_speed + self.norm_strength * (trill_speed * self.get_normalisation_coefficient(transition, ratios))
        return normalised_speed

    def inverse_normalise_by_session_index(self, transition, normalised_trill_speed, session_index: int):
        denormalised_speed = normalised_trill_speed / (1 - self.norm_strength * (1 - self.get_normalisation_coefficient(transition, self.ratios[session_index])))
        return denormalised_speed
    
    def inverse_normalise(self, transition, normalised_trill_speed, anchor_speeds):
        ratios = self.calculate_ratios_with_means(anchor_speeds, self.means)
        denormalised_speed = normalised_trill_speed / (1 - self.norm_strength * (1 - self.get_normalisation_coefficient(transition, ratios)))
        return denormalised_speed

def main():
    parser = argparse.ArgumentParser(
                    prog='Session Normalisation Tool',
                    description='Tool designed to normalise trill speeds between sessions and players from batch-analysed csv files')
    parser.add_argument('--csvs', type=str, help="Path to the folder of batch-analysed .csv files")
    parser.add_argument('-o', type=str, help="Will store all the outputs into one csv file with this name")
    parser.add_argument("--strength", type=float, help="Specify the normalisation strength")
    parser.add_argument("--norm_style", type=str, choices=['additive', 'multiplicative'], default='multiplicative')
    args = parser.parse_args()
    if (not os.path.isdir(args.csvs)):
        print("--csvs flag must be set to a directory containing .csv files!")
        input()
        exit()

    # Read all the anchors from every csv into dict {filename: [(encoding1, encoding2, speed)]}
    normaliser = None
    if args.norm_style == 'additive':
        normaliser = AdditiveAnchorBasedNormaliser(args.csvs, args.strength, encoding.ExpertFeatureNumberOfFingersExtractor())
    elif args.norm_style == 'multiplicative':
        normaliser = MultiplicativeAnchorBasedNormaliser(args.csvs, args.strength, encoding.ExpertFeatureNumberOfFingersExtractor())
    else:
        raise NotImplementedError()

    new_csv = []
    for i, filename in enumerate(normaliser.file_to_anchors_dict):
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
                normalised_speed = normaliser.normalise(transition, speed, i)
                # print(f"{filename} - {name1}, {name2}\nOG: {speed}, adjustment: {overall_adjustment}, new: {normalised_speed}")
                new_csv.append([filename, cluster, midi1, name1, encoding1, midi2, name2, encoding2, normalised_speed])
    
    with open(args.o, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["Filename", "Cluster", "Midi 1 (transposed as written for TS)", "Fingering 1 Name", "Fingering 1 Encoding", "Midi 2 (transposed as written for TS)", "Fingering 2 Name", "Fingering 2 Encoding", "(Normalised) Trill Speed", f"Strength {args.strength}"])
        for row in new_csv:
            writer.writerow(row)

if __name__ == "__main__":
    main()
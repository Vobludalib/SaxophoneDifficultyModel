import csv
from encoding import Fingering
import encoding
import argparse
import os
import numpy as np
import scipy.special
import json

# Takes seperate batch_analysed csvs (each corresponding to one session) and generates one big CSV of trills
# normalised using the anchor intervals

# ANCHORS ARE TAKEN AS BEING CLUSTER -1

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

def load_anchors_from_directory(directory_path) -> dict:
    file_to_anchors_dict = {}
    for file in os.listdir(directory_path):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            full_path = os.path.join(directory_path, filename)
            file_to_anchors_dict[filename] = load_anchors_from_file(full_path)

    return file_to_anchors_dict

def get_euclidean_distance(features1, features2):
        return np.sum([((features1[i] - features2[i]) ** 2) ** 0.5 for i in range(features1.shape[0])])

def calculate_anchor_speeds(file_to_anchors_dict) -> tuple[list, list]:
    """
    Return order is: list of transition objects, speeds.
    Speeds is a 2d list, row = session, column = anchor transition
    """

    anchor_intervals_sorted = []
    checked = False
    for filename in file_to_anchors_dict:
        file_to_anchors_dict[filename].sort(key = lambda tup: (tup[0].midi, tup[1].midi))
        if not checked:
            for tup in file_to_anchors_dict[filename]:
                anchor_intervals_sorted.append((tup[0], tup[1]))
            checked = True

    speeds = np.zeros((len(file_to_anchors_dict), len(anchor_intervals_sorted)), dtype=float)
    for row, filename in enumerate(file_to_anchors_dict):
        for column, interval_tup in enumerate(file_to_anchors_dict[filename]):
            speeds[row, column] = interval_tup[2]

    return anchor_intervals_sorted, speeds

def calculate_difference_to_mean(speeds, means):
    differences = np.zeros((speeds.shape[0], speeds.shape[1]), dtype=float)
    for row in range(speeds.shape[0]):
        for column in range(speeds.shape[1]):
            differences[row, column] = means[column] - speeds[row, column]

    return differences

def get_anchor_interval_features(anchor_intervals):
    anchor_features = []
    for tup in anchor_intervals:
        anchor_features.append(encoding.generate_transition_features(tup))
    anchor_features = np.asarray(anchor_features)
    return anchor_features

def normalise_transition(transition: encoding.Transition, transition_speed, session_index, anchor_features, differences, norm_strength=0.2):
    interval_features = encoding.generate_transition_features(transition)
    distances_to_anchors = np.asarray([get_euclidean_distance(interval_features, anchor_features[j]) for j in range(anchor_features.shape[0])])
    multiples = np.asarray(scipy.special.softmax(distances_to_anchors))
    overall_adjustment = np.sum(np.multiply(multiples, differences[session_index])) * norm_strength
    normalised_speed = transition_speed + overall_adjustment

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
    
    file_to_anchors_dict = load_anchors_from_directory(args.csvs)

    anchor_intervals_sorted, speeds = calculate_anchor_speeds(file_to_anchors_dict)
    print(f"Here is to order in which orders appear in the following matrices (columns):")
    print(anchor_intervals_sorted)
    print(f"Full array of anchor interval speeds (row = session, column = anchor):")
    print(speeds)
    print(f"Here are their averages over all sessions:")
    averages = np.mean(speeds, axis=0)
    print(averages)

    differences = calculate_difference_to_mean(speeds, averages)

    print(f"Here are the differences from the mean (negative means that speed was higher than mean):")
    print(differences)

    print(f"Here are the features of the anchor intervals:")
    anchor_features = get_anchor_interval_features(anchor_intervals_sorted)
    print(anchor_features)

    new_csv = []
    for i, filename in enumerate(file_to_anchors_dict):
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
                normalised_speed = normalise_transition(transition, speed, i, anchor_features, differences, args.strength)
                # print(f"{filename} - {name1}, {name2}\nOG: {speed}, adjustment: {overall_adjustment}, new: {normalised_speed}")
                new_csv.append([filename, cluster, midi1, name1, encoding1, midi2, name2, encoding2, normalised_speed])
    
    with open(args.o, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["Filename", "Cluster", "Midi 1 (transposed as written for TS)", "Fingering 1 Name", "Fingering 1 Encoding", "Midi 2 (transposed as written for TS)", "Fingering 2 Name", "Fingering 2 Encoding", "(Normalised) Trill Speed"])
        for row in new_csv:
            writer.writerow(row)

if __name__ == "__main__":
    main()
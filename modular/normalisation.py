import csv
from encoding import Fingering
import encoding
import argparse
import os
import numpy as np
import scipy.special

# Takes seperate batch_analysed csvs (each corresponding to one session) and generates one big CSV of trills
# normalised using the anchor intervals

# ANCHORS ARE TAKEN AS BEING CLUSTER -1

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
    
    file_to_anchors_dict = {}
    for file in os.listdir(args.csvs):
        filename = os.fsdecode(file)
        if filename.endswith('.csv'):
            full_path = os.path.join(args.csvs, filename)
            with open(full_path, 'r') as csvf:
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
                        if file_to_anchors_dict.get(filename, 0) == 0:
                            file_to_anchors_dict[filename] = [(Fingering(midi1, name1, encoding1), Fingering(midi2, name2, encoding2), speed)]
                        else:
                            file_to_anchors_dict[filename].append((Fingering(midi1, name1, encoding1), Fingering(midi2, name2, encoding2), speed))
    
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

    print(f"Here is to order in which orders appear in the following matrices (columns):")
    print(anchor_intervals_sorted)
    print(f"Full array of anchor interval speeds (row = session, column = anchor):")
    print(speeds)
    print(f"Here are their averages over all sessions:")
    averages = np.mean(speeds, axis=0)
    print(averages)

    differences = np.zeros((len(file_to_anchors_dict), len(anchor_intervals_sorted)), dtype=float)
    for row in range(speeds.shape[0]):
        for column in range(speeds.shape[1]):
            differences[row, column] = averages[column] - speeds[row, column]

    print(f"Here are the differences from the mean (negative means that speed was higher than mean):")
    print(differences)

    print(f"Here are the features of the anchor intervals:")
    anchor_features = []
    for tup in anchor_intervals_sorted:
        anchor_features.append(encoding.generate_transition_features(tup))
    anchor_features = np.asarray(anchor_features)
    print(anchor_features)

    def get_euclidean_distance(features1, features2):
        return np.sum([((features1[i] - features2[i]) ** 2) ** 0.5 for i in range(features1.shape[0])])

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
                interval_features = encoding.generate_transition_features(encoding.Transition(fingering1, fingering2))
                distances_to_anchors = np.asarray([get_euclidean_distance(interval_features, anchor_features[j]) for j in range(anchor_features.shape[0])])
                multiples = np.asarray(scipy.special.softmax(distances_to_anchors))
                overall_adjustment = np.sum(np.multiply(multiples, differences[i])) * args.strength
                normalised_speed = speed + overall_adjustment
                # print(f"{filename} - {name1}, {name2}\nOG: {speed}, adjustment: {overall_adjustment}, new: {normalised_speed}")
                new_csv.append([filename, cluster, midi1, name1, encoding1, midi2, name2, encoding2, normalised_speed])
    
    with open(args.o, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["Filename", "Cluster", "Midi 1 (transposed as written for TS)", "Fingering 1 Name", "Fingering 1 Encoding", "Midi 2 (transposed as written for TS)", "Fingering 2 Name", "Fingering 2 Encoding", "(Normalised) Trill Speed"])
        for row in new_csv:
            writer.writerow(row)

if __name__ == "__main__":
    main()
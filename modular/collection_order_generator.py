import argparse

import scipy.special
import encoding
import itertools
import random
import csv
import scipy
import numpy as np
import sklearn.cluster as skc

def generate_interval_difficulty_approx(interval):
    features = encoding.generate_interval_features(interval)[1]
    return (features[1] - features[0]) * 10 + (10 if features[0] < 6.1 else 0) + (10 if features[1] > 85 else 0) + features[8] + features[9]

def generate_cluster_difficulty_approx(clusters_dict):
    difficulties = []
    for key in clusters_dict:
        cluster_difficulty = 0
        i = 0
        for interval in clusters_dict[key]:
            # Simple difficulty generation for extreme jumps
            cluster_difficulty += generate_interval_difficulty_approx(interval)
            i += 1
        
        cluster_difficulty = cluster_difficulty / i
        difficulties.append(cluster_difficulty)
        i = 0

    return difficulties

def generate_sessions(number_of_intervals_per_sessions, clusters_dict, anchor_intervals, number_of_transitions, output_file, seed=10):
    sessions = []
    random.seed(seed)
    np.random.seed(seed)

    cluster_difficulties = np.asarray(generate_cluster_difficulty_approx(clusters_dict))
    softmax_difficulties = scipy.special.softmax(cluster_difficulties)
    inverse_softmax_difficulties = scipy.special.softmax(-cluster_difficulties)

    total_non_anchors_covered = 0
    cluster_label = 0
    anchor_cluster = []
    for anchor_interval in anchor_intervals:
            anchor_cluster.append(anchor_interval)

    while total_non_anchors_covered < number_of_transitions:
        newSession = [(-1, anchor_cluster)]
        number_of_intervals_in_session = len(anchor_cluster)

        while number_of_intervals_in_session < number_of_intervals_per_sessions and len(clusters_dict) > 0:
                cluster_difficulties = np.asarray(generate_cluster_difficulty_approx(clusters_dict))
                softmax_difficulties = scipy.special.softmax(cluster_difficulties)
                inverse_softmax_difficulties = scipy.special.softmax(-cluster_difficulties)

                sample = 0
                keys = list(clusters_dict.keys())
                # Beginning we sample from easy intervals
                if number_of_intervals_in_session < number_of_intervals_per_sessions * 0.25:
                    sample = np.random.choice(keys, size=1, p=inverse_softmax_difficulties)[0]
                elif number_of_intervals_in_session >= number_of_intervals_per_sessions * 0.25 and number_of_intervals_in_session <= number_of_intervals_per_sessions * 0.75:
                    sample = np.random.choice(keys, size=1, p=softmax_difficulties)[0]
                # Default weighting by easy
                else:
                    sample = np.random.choice(keys, size=1, p=softmax_difficulties)[0]

                newSession.append((cluster_label, clusters_dict[sample]))
                total_non_anchors_covered += len(clusters_dict[sample])
                number_of_intervals_in_session += len(clusters_dict[sample])
                cluster_label += 1

                del clusters_dict[sample]

        sessions.append(newSession)
        
        newSession = []
        number_of_intervals_in_session = 0

    for session in sessions:
        intervals = session[1]
        for cluster_i, cluster in enumerate(session):
            if cluster_i == 0: continue
            for interval in cluster[1]:
                interval_encodings = [interval[0].generate_encoding(), interval[1].generate_encoding()]
                for anchor_int in anchor_intervals:
                    if anchor_int[0].generate_encoding() in interval_encodings and anchor_int[1].generate_encoding() in interval_encodings:
                        cluster[1].remove(interval)

    with open(output_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Session", "Cluster", "Note 1 midi", "Note 1 name", "Note 1 encoding", "Note 2 midi", "Note 2 encoding", "Note 2 encoding", f"SEED:{seed}"])
        for i, session in enumerate(sessions):
            all_session_transitions = []
            for cluster_tuple in session:
                cluster_id = cluster_tuple[0]
                cluster = cluster_tuple[1]
                for transition in cluster:
                    all_session_transitions.append((cluster_id, transition[0], transition[1]))
                
            all_sessions_trans_dict = {}
            j = 0
            for tup in all_session_transitions:
                all_sessions_trans_dict[j] = tup
                j += 1

            j = 0
            while len(all_sessions_trans_dict) > 0:
                interval_difficulties = np.asarray([generate_interval_difficulty_approx((all_sessions_trans_dict[key][1], all_sessions_trans_dict[key][2])) for key in all_sessions_trans_dict.keys()])
                softmax_difficulties = np.asarray((scipy.special.softmax(interval_difficulties) + 0.3)) # 0.3 is 'temp' for softmax to make sure it's not only hard ones in the middle 
                softmax_difficulties = softmax_difficulties / np.sum(softmax_difficulties)
                inverse_softmax_difficulties = np.asarray(scipy.special.softmax(-interval_difficulties) + 0.3)
                inverse_softmax_difficulties = inverse_softmax_difficulties / np.sum(inverse_softmax_difficulties)

                sample = 0
                keys = list(all_sessions_trans_dict.keys())
                # Beginning we sample from easy intervals
                if j < len(all_session_transitions) * 0.25:
                    sample = np.random.choice(keys, size=1, p=inverse_softmax_difficulties)[0]
                elif j >= len(all_session_transitions) * 0.25 and number_of_intervals_in_session <= len(all_session_transitions) * 0.75:
                    sample = np.random.choice(keys, size=1, p=softmax_difficulties)[0]
                # Default weighting by easy
                else:
                    sample = np.random.choice(keys, size=1, p=softmax_difficulties)[0]

                trans_triple = all_sessions_trans_dict[sample]
                writer.writerow([i, trans_triple[0], trans_triple[1].midi, trans_triple[1].name, trans_triple[1].generate_encoding(), trans_triple[2].midi, trans_triple[2].name, trans_triple[2].generate_encoding()])
                del all_sessions_trans_dict[sample]

            # random.shuffle(all_session_transitions)
            # for trans_triple in all_session_transitions:
            #     writer.writerow([i, trans_triple[0], trans_triple[1].midi, trans_triple[1].name, trans_triple[1].generate_encoding(), trans_triple[2].midi, trans_triple[2].name, trans_triple[2].generate_encoding()])

def get_anchor_intervals(all_transitions, path_to_anchors):
    anchors_encodings = []
    with open(path_to_anchors, 'r') as f:
        for line in f:
            encoding_pair = line.strip().split(sep=',')
            anchors_encodings.append(encoding_pair)

    anchors = []
    for transition in all_transitions:
        encodings = [transition[0].generate_encoding(), transition[1].generate_encoding()]
        if encodings in anchors_encodings:
            anchors.append(transition)

    return anchors

def generate_interval_clusters(fingerings, number_of_notes_per_cluster = 5, print_debug=False):
    all_transitions = itertools.combinations(fingerings, 2)
    encoding_feature_pairs = []
    for transition in all_transitions:
        if print_debug: print(f"Going from {transition[0].name} to {transition[1].name}")
        name, features = encoding.generate_interval_features(transition)
        encoding_feature_pairs.append(([transition[0], transition[1]], features))
    
    # with this cluster_amount, you get clusters of 5 fingerings that are similar
    clusters_dict = {}
    cluster_amount = int(len(encoding_feature_pairs) / number_of_notes_per_cluster)
    _, labels, _ = skc.k_means(n_clusters=cluster_amount, X=np.asarray([pair[1] for pair in encoding_feature_pairs]))
    for label in range(cluster_amount):
        if print_debug: print(f"=== PROCESSING LABEL {label} ===")
        for index, elem in enumerate(encoding_feature_pairs):
            if labels[index] == label:
                if print_debug: print(f"{elem[0]} with feat {elem[1]}")
                if label not in clusters_dict:
                    clusters_dict[label] = [elem[0]]
                else:
                    clusters_dict[label].append(elem[0])

    if print_debug: print(f"In total there are {len(encoding_feature_pairs)} unique trills using default fingerings on TS")

    return clusters_dict

def main():
    parser = argparse.ArgumentParser(
                    prog='Generate trills order for data collection')
    parser.add_argument('--noc', type=int, help="Prefered number of clusters per recording session, mutually incompatible with noi and nos")
    parser.add_argument('--noi', type=int, help="Prefered number of intervals per recording session, mutually incompatible with noc and nos")
    parser.add_argument('--nos', type=int, help="Prefered number of recording sessions, mutually incompatible with noc and noi")
    parser.add_argument('--out', type=str, help="Filepath to where to store output. Default is sessions.csv", default="sessions.csv")
    parser.add_argument('--seed', type=int, help="Seed used when randomly generating the sessions", default="10")
    parser.add_argument('--anchors', type=str, help="Path to file with pairs of encodings representing anchor intervals. Each line should be in the form:\nENCODING1,ENCODING2")
    args = parser.parse_args()

    fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")
    all_transitions = list(itertools.combinations(fingerings, 2))
    number_of_transitions = len(all_transitions)
    clusters_dict = generate_interval_clusters(fingerings)
    args.anchors = "/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/anchor_intervals.txt"

    # GET ANCHOR INTERVALS
    anchor_intervals = get_anchor_intervals(all_transitions, args.anchors)
    number_of_anchor_intervals = len(anchor_intervals)

    if (args.nos is None and args.noc is None and args.noi is None):
        args.noi = 50

    if (args.nos is not None):
        print("Prioritising session breakdown")
        # NOT ACTUALLY BREAKING DOWN EVENLY - figure this out
        transitions_per_session = number_of_transitions//args.nos
        generate_sessions(transitions_per_session, clusters_dict, anchor_intervals, number_of_transitions, args.out, args.seed)
    elif (args.noc is not None):
        print("Prioritising cluster breakdown")
        transitions_per_session = number_of_transitions//args.noc
        generate_sessions(transitions_per_session, clusters_dict, anchor_intervals, number_of_transitions, args.out, args.seed)
    elif (args.noi is not None):
        print("Prioritising intervals breakdown")
        if args.noi <= number_of_anchor_intervals:
            print(f"Unable to breakdown into minimum {args.noi} interval sessions, given that there are {number_of_anchor_intervals} anchor intervals")

        generate_sessions(args.noi, clusters_dict, anchor_intervals, number_of_transitions, args.out, args.seed)
        


if __name__ == '__main__':
    main()
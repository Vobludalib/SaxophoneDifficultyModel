import argparse

import scipy.special
import encoding
import itertools
import random
import csv
import scipy
import numpy as np

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
        writer.writerow(["Session", "Cluster", "Note 1 midi", "Note 1 name", "Note 1 encoding", "Note 2 midi", "Note 2 encoding", "Note 2 encoding"])
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

# FOR NOW HARDCODED
def get_anchor_intervals(all_transitions):
    anchors = []
    for transition in all_transitions:
        encodings = [transition[0].generate_encoding(), transition[1].generate_encoding()]
        print(f"{transition[0].name} {transition[1].name}")
        print(f"{encodings}")
        if "1000100000000_0000000000" in encodings and "1010110000000_0000000000" in encodings:
            anchors.append(transition)
        elif "1010110000000_1000000000" in encodings and "1010110000000_1100110000" in encodings:
            anchors.append(transition)
        elif "0010110000000_1100100000" in encodings and "1010110000000_1100100000" in encodings:
            anchors.append(transition)
        elif "1010100000000_0000000000" in encodings and "1000000000011_0000000100" in encodings:
            anchors.append(transition)
        elif "0010110000000_1100101000" in encodings and "0010110000000_1000000000" in encodings:
            anchors.append(transition)
        elif "0010110000000_0000000000" in encodings and "0010111000000_0000000000" in encodings:
            anchors.append(transition)

    return anchors

# TODO: Encode anchor intervals (those that appear the most and or not in the same clusters)
def main():
    parser = argparse.ArgumentParser(
                    prog='Generate trills order for data collection')
    parser.add_argument('--noc', type=int, help="Prefered number of clusters per recording session, mutually incompatible with noi and nos")
    parser.add_argument('--noi', type=int, help="Prefered number of intervals per recording session, mutually incompatible with noc and nos")
    parser.add_argument('--nos', type=int, help="Prefered number of recording sessions, mutually incompatible with noc and noi")
    parser.add_argument('--out', type=str, help="Filepath to where to store output. Default is sessions.csv", default="sessions.csv")
    parser.add_argument('--seed', type=str, help="Seed used when randomly generating the sessions")
    args = parser.parse_args()

    fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")
    all_transitions = list(itertools.combinations(fingerings, 2))
    number_of_transitions = len(all_transitions)
    clusters_dict = encoding.generate_interval_clusters(fingerings)

    # GET ANCHOR INTERVALS
    anchor_intervals = get_anchor_intervals(all_transitions)
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
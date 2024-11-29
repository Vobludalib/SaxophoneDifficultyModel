import argparse
import encoding
import itertools
import random
import csv

def generate_sessions(number_of_intervals_per_sessions, clusters_dict, anchor_intervals, number_of_transitions, output_file):
    sessions = []

    randomise_cluster_order = random.sample(list(clusters_dict.keys()), len(list(clusters_dict.keys())))

    total_non_anchors_covered = 0
    cluster_label = 0
    anchor_cluster = []
    for anchor_interval in anchor_intervals:
            anchor_cluster.append(anchor_interval)

    while total_non_anchors_covered < number_of_transitions:
        newSession = [(-1, anchor_cluster)]
        number_of_intervals_in_session = len(anchor_cluster)

        while number_of_intervals_in_session < number_of_intervals_per_sessions:
                if (cluster_label not in clusters_dict):
                    sessions.append(newSession)
                    break
                else:
                    newSession.append((randomise_cluster_order[cluster_label], clusters_dict[randomise_cluster_order[cluster_label]]))
                    total_non_anchors_covered += len(clusters_dict[randomise_cluster_order[cluster_label]])
                    number_of_intervals_in_session += len(clusters_dict[randomise_cluster_order[cluster_label]])
                    cluster_label += 1

        sessions.append(newSession)
        
        newSession = []
        number_of_intervals_in_session = 0

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
                
            random.shuffle(all_session_transitions)
            for trans_triple in all_session_transitions:
                writer.writerow([i, trans_triple[0], trans_triple[1].midi, trans_triple[1].name, trans_triple[1].generate_encoding(), trans_triple[2].midi, trans_triple[2].name, trans_triple[2].generate_encoding()])


# TODO: Encode anchor intervals (those that appear the most and or not in the same clusters)
def main():
    parser = argparse.ArgumentParser(
                    prog='Generate trills order for data collection')
    parser.add_argument('--noc', type=int, help="Minimum number of clusters per recording session")
    parser.add_argument('--noi', type=int, help="Minimum number of intervals per recording session")
    parser.add_argument('--nos', type=int, help="Number of recording sessions")
    parser.add_argument('--out', type=str, help="Where to store output", default="sessions.csv")
    args = parser.parse_args()

    fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/encodings.txt")
    all_transitions = list(itertools.combinations(fingerings, 2))
    number_of_transitions = len(all_transitions)
    clusters_dict = encoding.generate_interval_clusters(fingerings)

    # GET ANCHOR INTERVALS
    anchor_intervals = all_transitions[0:8]
    number_of_anchor_intervals = 8

    if (args.nos is None and args.noc is None and args.noi is None):
        args.noi = 50

    if (args.nos is not None):
        print("Prioritising session breakdown")
        # NOT ACTUALLY BREAKING DOWN EVENLY - figure this out
        transitions_per_session = number_of_transitions//args.nos
        generate_sessions(transitions_per_session, clusters_dict, anchor_intervals, number_of_transitions, args.out)
    elif (args.noc is not None):
        print("Prioritising cluster breakdown")
        transitions_per_session = number_of_transitions//args.noc
        generate_sessions(transitions_per_session, clusters_dict, anchor_intervals, number_of_transitions, args.out)
    elif (args.noi is not None):
        print("Prioritising intervals breakdown")
        if args.noi <= number_of_anchor_intervals:
            print(f"Unable to breakdown into minimum {args.noi} interval sessions, given that there are {number_of_anchor_intervals} anchor intervals")

        generate_sessions(args.noi, clusters_dict, anchor_intervals, number_of_transitions, args.out)
        


if __name__ == '__main__':
    main()
import numpy as np
import crepe
import librosa
import matplotlib.pyplot as plt
import sklearn.cluster as skc
from scipy.io import wavfile
from scipy.signal.windows import gaussian
import os
import csv
import argparse
from pathlib import Path
from enum import Enum
from tqdm import tqdm

# Changing these values will change behaviour for segmentation of audio into subsegments
amount_of_splits = 4
splits_to_glue = 2
# Chaning this may help alleviate issues with note misdetection due to under/overblowing
outlier_threshold = 0.2

def get_note_distribution(notes):
    rnd = np.vectorize(lambda x: np.round(x))
    rounded = rnd(notes)
    rounded_notes, counts = np.unique(rounded, return_counts=True)
    total = np.sum(counts)
    return np.asarray([(rounded_notes[i] / total, counts[i]) for i in range(counts.shape[0])])

class ClusterIssue(Enum):
    EmptyCluster = 0
    ShortCluster = 1
    ClusterContains2OrMoreNotes = 2

def not_reasonable_clustering(notes1, notes2):
    if len(notes1) == 0 or len(notes2) == 0:
        return (True, ClusterIssue.EmptyCluster)
    elif (len(notes1) < len(notes2)*outlier_threshold or len(notes2) < len(notes1)*outlier_threshold):
        return (True, ClusterIssue.ShortCluster)
    else:       
        return (False, None)

def get_changes(f0_matrix):
    midi_vals = []
    for row in f0_matrix[1:]:
        if float(row[2]) > 0.8:
            midi_val = librosa.hz_to_midi(float(row[1]))
            midi_vals.append(midi_val)
    midi_vals = np.asarray(midi_vals)
    notes1 = []
    notes2 = []
    invalid_clustering = True
    had_problems = False
    while invalid_clustering:
        two_means_cluster = skc.KMeans(n_clusters=2).fit(midi_vals.reshape(-1, 1))
        notes1 = []
        notes2 = []
        for i in range(0, midi_vals.shape[0]):
            if two_means_cluster.labels_[i] == 0:
                notes1.append(midi_vals[i])
            else:
                notes2.append(midi_vals[i])
        notes1 = np.asarray(notes1)
        notes2 = np.asarray(notes2)
        note1_midis = [round(np.median(notes1))]
        note2_midis = [round(np.median(notes2))]

        invalid_clustering, reason = not_reasonable_clustering(notes1, notes2)
        if invalid_clustering:
            if reason == ClusterIssue.ShortCluster:
                print(f"WARNING THE ABOVE AUDIO HAD PROBLEMS WITH INITIAL CLUSTERING")
                had_problems = True
                if (len(notes1) < len(notes2)*outlier_threshold):
                    # notes1 is likely an outlier class
                    midi_vals = notes2
                elif (len(notes2) < len(notes1)*outlier_threshold):
                    midi_vals = notes1

    changes = 0
    last_midi_vals = 0
    time_spent_on_note = 0
    other_midi_vals = 0
    for midi_val in midi_vals:
        if last_midi_vals == 0:
            if round(midi_val) in note1_midis:
                last_midi_vals = note1_midis
                other_midi_vals = note2_midis
                continue
            elif round(midi_val) in note2_midis:
                last_midi_vals = note2_midis
                other_midi_vals = note1_midis
                continue
            else: continue
        
        if round(midi_val) in last_midi_vals:
            time_spent_on_note += 1
            continue   

        if round(midi_val) in other_midi_vals and time_spent_on_note > 1:
            changes += 1
            last_midi_vals, other_midi_vals = other_midi_vals, last_midi_vals
            time_spent_on_note = 0

    return changes, note1_midis, note2_midis, had_problems

def main():
    parser = argparse.ArgumentParser(
                    prog='Trill Parsing Tool',
                    description='Tool designed to automate trill speed analysis from audio files')
    parser.add_argument('-f', type=str, help="Path to the input file or directory")
    parser.add_argument('-r', action="store_true", help="If present, will iterate over all .wav files in given directory")
    parser.add_argument('-o', '--out', type=str, help="Path where to store the .csv file (e.g. /path/to/somewhere/trill.csv)")
    args = parser.parse_args()
    if (os.path.isdir(args.f) and not args.r):
        print(f"Invalid command line arguments! Given filepath is a directory and -r was not set!")
        return
    if (os.path.isfile(args.f) and not args.r):
        print(evaluate(args.f))
    elif (os.fsdecode(args.out) and os.fsdecode(args.out).endswith('.csv')):
            output = []
            problem_files = []
            output.append(["Filename", "Note 1 MIDI", "Note 2 MIDI", "Best Trill Speed"])
            for file in os.listdir(args.f):
                filename = os.fsdecode(file)
                if filename.endswith('.wav'):
                    print(f"=========== Handling file: {filename} ============")
                    path = os.path.join(args.f, filename)
                    note1_midi, note2_midi, trill_speed, _, is_possible_problem = evaluate(path)
                    if note1_midi <= note2_midi:
                        output.append([filename, note1_midi, note2_midi, trill_speed])
                    else:
                        output.append([filename, note2_midi, note1_midi, trill_speed])
                    
                    if is_possible_problem:
                        problem_files.append(filename)
                    
            with open(args.out, 'w') as csvFile:
                writer = csv.writer(csvFile, lineterminator='\n')
                writer.writerows(output)

            with open(args.out + '.log', 'w') as log:
                for problem in problem_files:
                    log.write(problem + '\n')
    else:
        print(f"Invalid arguments. See --help for usage.")
        exit()

def evaluate(f_path):
    f0_matrix = None

    if (Path(f_path).suffix == ".wav"):
        sr, audio = wavfile.read(f_path)
        time, frequency, confidence, _ = crepe.predict(audio, sr, viterbi=False)
        f0_matrix = np.array([time, frequency, confidence]).T

    # Allows for loading from a crepe .csv file - not used as of now, but useful if you want to rerun
    # this operation without having to do crepe prediction each time
    elif (Path(f_path).suffix == ".csv"):
        rows = []
        with open(f_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                rows.append(row)

        f0_matrix = np.asarray(rows)[1:]

    midi_splits = np.array_split(f0_matrix, amount_of_splits)

    split_trill_speeds = []
    is_possible_problem = False
    for i in range((amount_of_splits - splits_to_glue) + 1):
        glued = np.vstack(midi_splits[i:i+splits_to_glue])
        split_changes, note1_midi, note2_midi, had_problems = get_changes(glued)
        if not is_possible_problem and had_problems: is_possible_problem = True
        split_duration_in_s = float(glued[-1][0]) - float(glued[0][0])
        trill_speed = split_changes / split_duration_in_s / 2
        split_trill_speeds.append(trill_speed)

    trill_speeds = np.asarray(split_trill_speeds)
    best_split = np.argmax(split_trill_speeds)
    best_trill_speed = np.max(trill_speeds)
    
    return note1_midi[0] + 2, note2_midi[0] + 2, best_trill_speed, best_split, is_possible_problem

if __name__ == "__main__":
    main()
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

def get_changes(f0_matrix):
    midi_vals = []
    for row in f0_matrix[1:]:
        if float(row[2]) > 0.8:
            midi_val = librosa.hz_to_midi(float(row[1]))
            midi_vals.append(midi_val)
    midi_vals = np.asarray(midi_vals)
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
    note1_midi = round(np.median(notes1))
    note2_midi = round(np.median(notes2))

    changes = 0
    last_midi_val = 0
    time_spent_on_note = 0
    other_midi_val = 0
    for midi_val in midi_vals:
        if last_midi_val == 0:
            if round(midi_val) == note1_midi:
                last_midi_val = note1_midi
                other_midi_val = note2_midi
                continue
            elif round(midi_val) == note2_midi:
                last_midi_val = note2_midi
                other_midi_val = note1_midi
                continue
        
        if round(midi_val) == last_midi_val:
            time_spent_on_note += 1
            continue   

        if round(midi_val) == other_midi_val and time_spent_on_note > 2:
            changes += 1
            last_midi_val, other_midi_val = other_midi_val, last_midi_val
            time_spent_on_note = 0

    return changes, note1_midi, note2_midi

def main():
    parser = argparse.ArgumentParser(
                    prog='Trill Parsing Tool',
                    description='Tool designed to automate trill speed analysis from audio files')
    parser.add_argument('-f', type=str)
    parser.add_argument('-r', action="store_true", help="If present, will iterate over all .wav files in given directory")
    parser.add_argument('--console', action="store_true", help="If present, will print output to console instead of default file")
    args = parser.parse_args()
    if (os.path.isdir(args.f) and not args.r):
        print(f"Given filepath is a directory and -r was not set!")
        return
    if (os.path.isfile(args.f)):
        evaluate(args.f)
    if (os.path.isdir(args.f) and args.r):
        for file in os.listdir(args.f):
            filename = os.fsdecode(file)
            if filename.endswith('.wav'):
                print(f"=========== Handling file: {filename} ============")
                path = os.path.join(args.f, filename)
                evaluate(path, write_to_console=args.console)

def evaluate(f_path, write_to_console = False):
    f0_matrix = None

    if (Path(f_path).suffix == ".wav"):
        sr, audio = wavfile.read(f_path)
        time, frequency, confidence, _ = crepe.predict(audio, sr, viterbi=False)
        f0_matrix = np.array([time, frequency, confidence]).T

    elif (Path(f_path).suffix == ".csv"):
        rows = []
        with open(f_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                rows.append(row)

        f0_matrix = np.asarray(rows)[1:]

    if( not write_to_console ):
        path = Path(f_path)
        basename = path.stem
        dirname = os.path.dirname(f_path)

    amount_of_splits = 4
    splits_to_glue = 2
    midi_splits = np.array_split(f0_matrix, amount_of_splits)

    split_trill_speeds = []
    for i in range((amount_of_splits - splits_to_glue) + 1):
        glued = np.vstack(midi_splits[i:i+splits_to_glue])
        split_changes, note1_midi, note2_midi = get_changes(glued)
        split_duration_in_s = float(glued[-1][0]) - float(glued[0][0])
        trill_speed = split_changes / split_duration_in_s / 2
        split_trill_speeds.append(trill_speed)

    trill_speeds = np.asarray(split_trill_speeds)
    best_split = np.argmax(split_trill_speeds)
    best_trill_speed = np.max(trill_speeds)
    if (write_to_console):
        print(f"{best_trill_speed:.2f} - best trill achieved in periods a second\n^^^^ Achieved at split indexes {best_split}-{best_split + splits_to_glue - 1}")
        print(f"Trilling from {librosa.midi_to_note(note1_midi + 2)} to {librosa.midi_to_note(note2_midi + 2)}")
    else:
        with open(os.path.join(dirname, basename + ".txt"), 'w') as output_file:
            output_file.write(f"{best_trill_speed:.2f} - best trill achieved in periods a second\n^^^^ Achieved at split indexes {best_split}-{best_split + splits_to_glue - 1}\n")
            output_file.write(f"Trilling from {librosa.midi_to_note(note1_midi + 2)} to {librosa.midi_to_note(note2_midi + 2)}")

if __name__ == "__main__":
    main()
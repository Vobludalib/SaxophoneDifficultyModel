import music21
import sklearn.neighbors
import sklearn.neural_network
import encoding
import model
import fingering_prediction
import matplotlib.colors as mcolors
import numpy as np
import math
import sklearn
from scipy.signal.windows import hann

# TODO: Make this into a class, wrapping over FingeringPredictor and TrillSpeedModel

def load_xml_file(path):
    m21stream = music21.converter.parseFile(path)
    return m21stream

def get_offsets_and_durations(part):
    """
    Method for taking duration and offset values for notes and rests in a path. 
    Rests past the last note in the part are ignored.
    """
    time_data = []
    block_to_append = []
    for noteOrRest in part.flatten().notesAndRests:
        block_to_append.append((noteOrRest, noteOrRest.offset, noteOrRest.duration))        
        if type(noteOrRest) is music21.note.Note:
            time_data = time_data + block_to_append
            block_to_append = []

    return time_data

def offset_and_duration_to_wall_clock_time(offsets_and_durations, bpm=120):
    times = []
    bps = bpm/60
    for noteOrRest, offset, duration in offsets_and_durations:
        if type(noteOrRest) == music21.note.Note:
            time_onset = offset / bps
            time_offset = (offset + duration.quarterLength) / bps
            times.append((noteOrRest, time_onset, time_offset))

    return times

# COULD BE REFINED TO FACTOR IN THE ABILITY TO NOT PLAY TILL OFFSET, BUT CUT NOTE SHORT
def split_based_on_rests(onsets_and_offsets, reset_time=0.5):
    sequences = []
    sequence = []
    for i, (note, onset, offset) in enumerate(onsets_and_offsets):
        if i == 0:
            sequence.append((note, onset, offset))
        else:
            if onset - reset_time > onsets_and_offsets[i-1][2]:
                sequences.append(sequence)
                sequence = [(note, onset, offset)]
            else:
                sequence.append((note, onset, offset))

    if len(sequence) != 0:
        sequences.append(sequence)

    return sequences

def get_difficulties_of_sequence(sequence, difficulty_model):
    """
    For every note in the sequence, the difficulty is treated as a 'weighted' average of the percentages of maximum trill speed of the neighbours
    """
    difficulties = []

    if len(sequence) == 1:
        difficulties = [0]
        return difficulties

    # Special case for first and last note
    # First note:
    curr_onset = sequence[0][2]
    curr_offset = sequence[0][3]
    curr_fingering = sequence[0][0]
    next_onset = sequence[1][2]
    next_offset = sequence[1][3]
    next_fingering = sequence[1][0]

    curr_to_next_estimated_ts = (1 / (next_offset - curr_onset))

    curr_to_next_predicted_ts = model.predict_fingering_transition(difficulty_model, curr_fingering, next_fingering)

    curr_to_next_percent = (curr_to_next_estimated_ts / curr_to_next_predicted_ts) * 100

    difficulties.append(curr_to_next_percent)

    # Treat the rest (indexes 1 to n - 1)
    for i in range(len(sequence) - 2):
        central_index = i + 1
        previous_index = i
        next_index = i + 2
        prev_onset = sequence[previous_index][2]
        prev_fingering = sequence[previous_index][0]
        curr_onset = sequence[central_index][2]
        curr_offset = sequence[central_index][3]
        curr_fingering = sequence[central_index][0]
        next_onset = sequence[central_index][2]
        next_offset = sequence[next_index][3]
        next_fingering = sequence[next_index][0]

        prev_to_curr_estimated_ts = (1 / (curr_offset - prev_onset))
        curr_to_next_estimated_ts = (1 / (next_offset - curr_onset))

        prev_to_curr_predicted_ts = model.predict_fingering_transition(difficulty_model, prev_fingering, curr_fingering)
        curr_to_next_predicted_ts = model.predict_fingering_transition(difficulty_model, curr_fingering, next_fingering)

        prev_to_curr_percent = (prev_to_curr_estimated_ts / prev_to_curr_predicted_ts) * 100
        curr_to_next_percent = (curr_to_next_estimated_ts / curr_to_next_predicted_ts) * 100

        if prev_to_curr_percent > 100 or curr_to_next_percent > 100:
            difficulties.append(max(prev_to_curr_percent, curr_to_next_percent))
        else:
            difficulties.append(prev_to_curr_percent * 0.5 + curr_to_next_percent * 0.5)

    # First note:
    prev_onset = sequence[previous_index][2]
    prev_fingering = sequence[previous_index][0]
    curr_onset = sequence[central_index][2]
    curr_offset = sequence[central_index][3]
    curr_fingering = sequence[central_index][0]

    prev_to_curr_estimated_ts = (1 / (curr_offset - prev_onset))

    prev_to_curr_predicted_ts = model.predict_fingering_transition(difficulty_model, prev_fingering, curr_fingering)

    prev_to_curr_percent = (prev_to_curr_estimated_ts / prev_to_curr_predicted_ts) * 100

    difficulties.append(prev_to_curr_percent)

    return difficulties

def get_difficulties_of_sequence_smoothed(sequence, difficulty_model, slice_resolution=0.05, window_size=0.5):
    # Pregenerate the difficulty of each slice
    start_onset = sequence[0][2]
    end_offset = sequence[-1][3]

    transition_times = []
    for tup1, tup2 in zip(sequence[:-1], sequence[1:]):
        # When onsets don't match, due to a small pause, we take the middle of the offset and onset
        if tup1[3] != tup2[2]:
            transition_times.append((tup1[3] + tup2[2]) / 2)
        else:
            transition_times.append(tup1[3])

    transition_boundaries = []
    for time1, time2 in zip(transition_times[:-1], transition_times[1:]):
        transition_boundaries.append((time1 + time2) / 2)

    # To mitigate some float-based issues
    transition_boundaries = np.asarray([round(time, 4) for time in transition_boundaries])

    slice_times = []
    end_offset_to_ensure_slice_resolution = slice_resolution * math.ceil(end_offset/slice_resolution)
    for slice in np.linspace(start_onset, end_offset_to_ensure_slice_resolution, int((end_offset_to_ensure_slice_resolution-start_onset)/slice_resolution) + 1):
        slice_times.append(round(slice, 4))

    # Map each slice to a given an index that will correspond to the local difficulty in that slice
    # This local difficulty will be taken as the difficulty of the nearest transition
    slice_to_transition_binding = np.digitize(slice_times, transition_boundaries)

    transition_difficulties = []
    for tup1, tup2 in zip(sequence[:-1], sequence[1:]):
        curr_onset = tup1[2]
        next_offset = tup2[3]
        curr_fingering = tup1[0]
        next_fingering = tup2[0]

        curr_to_next_estimated_ts = (1 / (next_offset - curr_onset))

        curr_to_next_predicted_ts = model.predict_fingering_transition(difficulty_model, curr_fingering, next_fingering)

        curr_to_next_percent = (curr_to_next_estimated_ts / curr_to_next_predicted_ts) * 100

        transition_difficulties.append(curr_to_next_percent)

    slice_difficulties = []
    for transition_index in slice_to_transition_binding:
        slice_difficulties.append(transition_difficulties[transition_index])

    slice_difficulties = np.asarray(slice_difficulties)

    # slice_difficulties is now the difficulty value of the nearest transition for each slice
    if int(window_size/slice_resolution) % 2 == 0:
        window_size -= slice_resolution
    m = round(window_size/slice_resolution)
    window = hann(m, sym=True)

    # Here working with assumption that the window is of odd length as ensured by the above code
    amount_on_each_side = int((window.shape[0] - 1) / 2)

    difficulties = []
    # For each note, we take it's midpoint, find it's slice, and perform a windowed difficulty using the other nearby slices
    slice_times_len = len(slice_times)
    for tup in sequence:
        midpoint = (tup[3] - tup[2]) / 2 + tup[2]
        time_slice_of_midpoint = int(np.digitize([midpoint], slice_times)[0])
        left_most_slice = max(0, time_slice_of_midpoint - amount_on_each_side)
        right_most_slice = min(slice_times_len - 1, time_slice_of_midpoint + amount_on_each_side)
        amount_on_left_window = time_slice_of_midpoint - left_most_slice
        amount_on_right_window = right_most_slice - time_slice_of_midpoint
        window_actually_used = window[amount_on_each_side - amount_on_left_window:amount_on_each_side+amount_on_right_window]
        normalised_window = np.asarray(window_actually_used) / np.sum(window_actually_used)
        difficulties_in_range = slice_difficulties[left_most_slice:right_most_slice]
        difficulty = np.sum(np.multiply(difficulties_in_range, normalised_window))
        difficulties.append(difficulty)

    return difficulties

def color_map_difficulty(difficulty_value):
    """
    Maps values 0 - 100+ to a color.
    0 - 100 goes 
    """
    if difficulty_value >= 150:
        return "#FF0000"  # Red for values 100+
    
    # Define the color gradient from light green to dark orange
    colors = ["#000000", "#FF0000"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # Normalize value to the range 0-1
    norm_value = np.clip(difficulty_value / 100, 0, 1)
    
    # Convert normalized value to an RGB tuple and then to hex
    return mcolors.to_hex(cmap(norm_value))

def main():
    # Load data
    transitions_speed_dict = encoding.load_transitions_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/files/normalisation_csvs/ALL_DATA.csv")
    to_delete = []
    for key in transitions_speed_dict:
        if key.fingering1.midi == key.fingering2.midi:
            to_delete.append(key)

    for delete in to_delete:
        transitions_speed_dict.pop(delete, None)

    # Train model
    fe = encoding.ExpertFeatureNumberOfFingersExtractor(use_expert_weights=False, remove_midi=False)
    xs, ys = model.transitions_trill_dict_to_numpy_arrays(transitions_speed_dict, fe)
    mlp = model.TrillSpeedModel(fe, False)
    mlp.set_custom_training_data(xs, ys)
    mlp.train_model(sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, solver="lbfgs"))

    # Load midi_to_fingering dict
    fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")
    midi_to_fingerings_dict = {}
    for fingering in fingerings:
        if midi_to_fingerings_dict.get(fingering.midi, None) is None:
            midi_to_fingerings_dict[fingering.midi] = [fingering]
        else:
            midi_to_fingerings_dict[fingering.midi].append(fingering)

    stream = load_xml_file("/Users/slibricky/Desktop/Thesis/thesis/modular/files/Short_Segment-Tenor_Saxophone.mxl")
    p = stream.parts[0]

    offsets_and_durations = get_offsets_and_durations(p)

    # Set BPM for a quarter note
    bpm = 160
    times = offset_and_duration_to_wall_clock_time(offsets_and_durations, bpm)
    # print(times)

    # Time to 'reset' - i.e. if a pause of reset_time seconds is seen, the two note sequences are treated as independent
    reset_time = 0.5
    splits = split_based_on_rests(times, reset_time)
    # print([f"{i}: {splits[i]}" for i in range(len(splits))])

    fingering_predictor = fingering_prediction.FingeringPrediction(mlp, midi_to_fingerings_dict)

    predicted_splits = []
    for i, split in enumerate(splits): 
        print(f"Doing fingering prediction on sequence {i}")
        midi_values = [note[0].pitch.midi for note in split]
        _, predictions = fingering_predictor.predict_fingerings(midi_values)
        predicted_split = []
        for i, (note, onset, offset) in enumerate(split):
            predicted_split.append((predictions[i], note, onset, offset))
        
        predicted_splits.append(predicted_split)

    split_difficulties = []
    for split in predicted_splits:
        split_difficulties.append(get_difficulties_of_sequence(split, mlp))
        # split_difficulties.append(get_difficulties_of_sequence_smoothed(split, mlp))

    for split_index, split in enumerate(splits):
        for i, (note, _, _) in enumerate(split):
            print(split_difficulties[split_index][i])
            note.style.color = color_map_difficulty(split_difficulties[split_index][i])

    stream.write("musicxml", "/Users/slibricky/Desktop/Thesis/thesis/modular/files/Short_Segment_Annotated.mxl")

if __name__ == "__main__":
    main()
import music21
import encoding
import model
import fingering_prediction
import matplotlib.colors as mcolors
import numpy as np

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

# TODO: CHANGE TO SOME KIND OF WINDOW METHOD
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
    transitions_speed_dict = encoding.load_transitions_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/data.csv")
    to_delete = []
    for key in transitions_speed_dict:
        if key.fingering1.midi == key.fingering2.midi:
            to_delete.append(key)

    for delete in to_delete:
        transitions_speed_dict.pop(delete, None)

    # Train model
    xs, ys = model.transitions_trill_dict_to_numpy_arrays(transitions_speed_dict)
    mlp = model.fit_on_mlp(xs, ys)

    # Load midi_to_fingering dict
    fingerings = encoding.load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt")
    midi_to_fingerings_dict = {}
    for fingering in fingerings:
        if midi_to_fingerings_dict.get(fingering.midi, None) is None:
            midi_to_fingerings_dict[fingering.midi] = [fingering]
        else:
            midi_to_fingerings_dict[fingering.midi].append(fingering)

    stream = load_xml_file("/Users/slibricky/Desktop/Thesis/thesis/modular/files/DifficultyTest1-Tenor_Saxophone.mxl")
    p = stream.parts[0]
    # for n in p.flatten().notes:
    #     print(n.offset)
    #     n.style.color = "red"

    offsets_and_durations = get_offsets_and_durations(p)

    # Set BPM for a quarter note
    bpm = 200
    times = offset_and_duration_to_wall_clock_time(offsets_and_durations, bpm)
    # print(times)

    # Time to 'reset' - i.e. if a pause of reset_time seconds is seen, the two note sequences are treated as independent
    reset_time = 0.5
    splits = split_based_on_rests(times, reset_time)
    # print([f"{i}: {splits[i]}" for i in range(len(splits))])

    predicted_splits = []
    for i, split in enumerate(splits): 
        print(f"Doing fingering prediction on sequence {i}")
        midi_values = [note[0].pitch.midi for note in split]
        _, predictions = fingering_prediction.midi_to_fingering_prediction(midi_values, mlp, midi_to_fingerings_dict)
        predicted_split = []
        for i, (note, onset, offset) in enumerate(split):
            predicted_split.append((predictions[i], note, onset, offset))
        
        predicted_splits.append(predicted_split)

    split_difficulties = []
    for split in predicted_splits:
        split_difficulties.append(get_difficulties_of_sequence(split, mlp))

    for split_index, split in enumerate(splits):
        for i, (note, _, _) in enumerate(split):
            print(split_difficulties[split_index][i])
            note.style.color = color_map_difficulty(split_difficulties[split_index][i])

    stream.write("musicxml", "/Users/slibricky/Desktop/Thesis/thesis/modular/files/annotated.mxl")

if __name__ == "__main__":
    main()
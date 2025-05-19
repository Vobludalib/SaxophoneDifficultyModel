import argparse
import music21
import matplotlib.colors as mcolors
import numpy as np
import os
import random
import string
import saxophone_difficulty_model as difficulty_model

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

def color_map_difficulty(difficulty_value, easy_color, hard_color):

    if difficulty_value >= 100:
        difficulty_value = 100
    
    if difficulty_value <= 0:
        difficulty_value = 0
    
    colors = [easy_color, hard_color]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # Normalize value to the range 0-1
    norm_value = np.clip(difficulty_value / 100, 0, 1)
    
    # Convert normalized value to an RGB tuple and then to hex
    return mcolors.to_hex(cmap(norm_value))

def generate_random_string(len=10):
    letters = string.ascii_lowercase + "".join([str(i) for i in range (0, 9)])
    return ''.join(random.choice(letters) for i in range(len))

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', '--tempPath', type=str, help="Path to input .mxl file.", required=True)
        parser.add_argument('--bpm', type=int, help="Set quarter note BPM for difficulty estimation.", default=120)
        parser.add_argument('--easy_color', type=str, help="Set the color for easy notes", default='#000000')
        parser.add_argument('--hard_color', type=str, help="Set the color for hard notes", default='#FF0000')
        args, leftovers = parser.parse_known_args()

        stream = load_xml_file(args.input)
        p = stream.parts[0]

        offsets_and_durations = get_offsets_and_durations(p)

        # Set BPM for a quarter note
        bpm = args.bpm
        times = offset_and_duration_to_wall_clock_time(offsets_and_durations, bpm)

        # Time to 'reset' - i.e. if a pause of reset_time seconds is seen, the two note sequences are treated as independent
        reset_time = 0.5
        splits = split_based_on_rests(times, reset_time)

        try:
            split_difficulties = difficulty_model.evaluate_difficulty(splits, {'tempFilePath': args.input})
        except Exception as e:
            print(e)
            return

        for split_index, split in enumerate(splits):
            for i, (note, _, _) in enumerate(split):
                note.style.color = color_map_difficulty(split_difficulties[split_index][i], args.easy_color, args.hard_color)

        outputPath = os.path.join(str(os.path.dirname(args.input)), generate_random_string() + '.musicxml')
        
        stream.write('musicxml', outputPath)
        print(os.path.abspath(outputPath))
        return os.path.abspath(outputPath)
        

if __name__ == "__main__":
    main()
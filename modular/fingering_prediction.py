import encoding
import model
import sklearn
import pickle
import numpy as np

# TODO: Mess around with different distance functions
# TODO: Handle same-note transitions
def midi_to_fingering_prediction(midi_values: list[int], difficulty_model, midi_to_fingering_dict: dict):
    for midi_note in midi_values:
        key = midi_to_fingering_dict.get(midi_note, None)
        if key is None:
            raise Exception(f"Midi value {midi_note} has no fingerings")
        
    # Memory is in the form [fingering, shortest distance from start to this midi value played by this note, [list of fingerings on the shortest path]]
    memory = [[fingering, 0, []] for fingering in midi_to_fingering_dict[midi_values[0]]]

    # Add ending delimiter
    midi_values.append(-1)

    # Note, shortest is 'inverted', as shortest distance corresponds to highest trill speed
    pairs = list(zip(midi_values, midi_values[1:]))
    for i, pair in enumerate(pairs):

        # We hit end delimiter
        if pair[1] == -1:
            best_distance = -np.inf
            best_path = []
            best_i = 0
            for i, path in enumerate(memory):
                if path[1] > best_distance:
                    best_path = path[2]
                    best_distance = path[1]
                    best_i = 0

            best_path.append(memory[best_i][0])

            return best_distance, best_path 
        
        current_viable_fingerings = [memory_unit[0] for memory_unit in memory]
        fingerings2 = midi_to_fingering_dict[pair[1]]

        new_memory = []
        for fing1_index, fing_1 in enumerate(current_viable_fingerings):
            for fing2_index, fing_2 in enumerate(fingerings2):
                predicted_trill_speed = model.predict_fingering_transition(difficulty_model, fing_1, fing_2)
                if pair[0] == pair[1]:
                    # Handling same-note transitions as being 'forced' to stay on one fingering to prevent
                    # changing same-note fingerings during repeated appearances
                    if fing_1 != fing_2:
                        predicted_trill_speed = 0.00001
                    else:
                        predicted_trill_speed = 10
                if fing1_index == 0:
                    new_memory.append([fing_2, memory[fing1_index][1] + predicted_trill_speed, memory[fing1_index][2].copy()])
                    new_memory[fing2_index][2].append(fing_1)
                else:
                    # If we found a better path to this 2nd fingering
                    if new_memory[fing2_index][1] < memory[fing1_index][1] + predicted_trill_speed:
                        new_memory[fing2_index][1] = memory[fing1_index][1] + predicted_trill_speed
                        new_memory[fing2_index][2][-1] = fing_1

        memory = new_memory

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

    # Do estimation
    output = midi_to_fingering_prediction([75, 82, 79, 78, 78, 77, 78], mlp, midi_to_fingerings_dict)
    print(f"BEST OUTPUT:")
    print(output[1])

if __name__ == '__main__':
    main()
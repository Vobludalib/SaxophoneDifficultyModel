import sklearn.neural_network
import encoding
import model
import sklearn
import numpy as np
import os

# TODO: Refactor into a class that does this and works over the TrillSpeedModel class
class FingeringPrediction():
    def __init__(self, trill_speed_model: model.TrillSpeedModel, midi_to_fingerings_dict: dict):
        self.model = trill_speed_model
        self.midi_to_fingerings_dict = midi_to_fingerings_dict

    def predict_fingerings(self, midi_values: list[int]):
        prev_midi = -1
        guide_index = -1
        deduplicated_midi_values = []
        duplication_guide = []
        for i, midi_note in enumerate(midi_values):
            key = self.midi_to_fingerings_dict.get(midi_note, None)
            if key is None:
                raise Exception(f"Midi value {midi_note} has no fingerings")
            
            # Preprocess repeated notes of the same value by removing duplicates, and then adding again at the end
            if prev_midi != midi_note:
                guide_index += 1
                deduplicated_midi_values.append(midi_note)
                prev_midi = midi_note
                duplication_guide.append(guide_index)
            else:
                duplication_guide.append(guide_index)

        midi_values = deduplicated_midi_values
            
        # Memory is in the form [fingering, shortest distance from start to this midi value played by this note, [list of fingerings on the shortest path]]
        memory = [[fingering, 0, []] for fingering in self.midi_to_fingerings_dict[midi_values[0]]]

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

                # Perform duplication to counter the initial deduplication
                reduplicated_best_path = []
                for index in duplication_guide:
                    reduplicated_best_path.append(best_path[index])

                return best_distance, reduplicated_best_path 
            
            current_viable_fingerings = [memory_unit[0] for memory_unit in memory]
            fingerings2 = self.midi_to_fingerings_dict[pair[1]]

            new_memory = []
            for fing1_index, fing_1 in enumerate(current_viable_fingerings):
                for fing2_index, fing_2 in enumerate(fingerings2):
                    predicted_trill_speed = self.model.predict_transitions([encoding.Transition(fing_1, fing_2)])[0]
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
    #TODO: CONVERT INTO CL tool
    # Load data
    transitions_speed_dict = encoding.load_transitions_from_file("PATH/TO/DATA.csv")
    
    # Remove same-note transitions from the training data
    to_delete = []
    for key in transitions_speed_dict:
        if key.fingering1.midi == key.fingering2.midi:
            to_delete.append(key)

    for delete in to_delete:
        transitions_speed_dict.pop(delete, None)

    # Train model
    fe = encoding.RawFeatureExtractor()
    xs, ys = model.transitions_trill_dict_to_numpy_arrays(transitions_speed_dict, feature_extractor=fe)
    mlp = model.TrillSpeedModel(fe, perform_only_infilling=False)
    # Doing this to load infilling data
    mlp.load_data_from_csv("PATH/TO/DATA.csv")
    mlp.set_custom_training_data(xs, ys)
    np.random.seed(10)
    mlp.train_model(sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, solver='lbfgs'))

    # Load midi_to_fingering dict
    fingerings = encoding.load_fingerings_from_file(os.path.join(".", "documentation", "encodings.txt"))
    midi_to_fingerings_dict = {}
    for fingering in fingerings:
        if midi_to_fingerings_dict.get(fingering.midi, None) is None:
            midi_to_fingerings_dict[fingering.midi] = [fingering]
        else:
            midi_to_fingerings_dict[fingering.midi].append(fingering)

    fing_pred = FingeringPrediction(mlp, midi_to_fingerings_dict)

    # Do estimation
    output = fing_pred.predict_fingerings([75, 82, 79, 78, 78, 77, 78])
    print(f"BEST OUTPUT:")
    print(output[1])
    print(output[0])

if __name__ == '__main__':
    main()
import dependencies.difficulty_estimator as de
import dependencies.encoding as e
import dependencies.fingering_prediction as fp
import dependencies.model as m

import music21

import pickle
import os
import sklearn

def train_model(path_to_data):
    pass

def save_model(model):
    pass

def try_load_model(model_path):
    # See if we can load a model
    if os.path.isfile(model_path):
        try:
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            return model
        except:
            return None

    return None

def train_model(data_path):
    transitions_speed_dict = e.load_transitions_from_file(data_path)
    to_delete = []
    for key in transitions_speed_dict:
        if key.fingering1.midi == key.fingering2.midi:
            to_delete.append(key)

    for delete in to_delete:
        transitions_speed_dict.pop(delete, None)

    # Train model
    fe = e.ExpertFeatureNumberOfFingersExtractor(use_expert_weights=False, remove_midi=False)
    xs, ys = m.transitions_trill_dict_to_numpy_arrays(transitions_speed_dict, fe)
    mlp = m.TrillSpeedModel(fe, False)
    mlp.set_custom_training_data(xs, ys)
    mlp.train_model(sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50,), max_iter=100000, solver="lbfgs"))

    return mlp

def save_model(model, model_path):
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

def evaluate_difficulty(
        phrases: list[list[tuple[music21.note.Note, float, float]]], # Each element of the outer list is a musical phrase (inner list). Each inner list contains tuples of music21 Notes, the onset, and offset time respectively.
        additional_info: dict = dict()
        ):
    
    tempFileDir = os.path.dirname(additional_info['tempFilePath'])
    curr_dir = os.path.join(os.path.join(tempFileDir, '..'))
    model_path = os.path.join(curr_dir, 'model', 'model.pickle')
    data_path = os.path.join(curr_dir, 'model', 'Processed_Data.csv')
    encodings_path = os.path.join(curr_dir, 'model', 'encodings.txt')

    model = try_load_model(model_path)
    if model is None:
        model = train_model(data_path)
        save_model(model, model_path)

    # Load midi_to_fingering dict
    fingerings = e.load_fingerings_from_file(encodings_path)

    midi_to_fingerings_dict = {}
    for fingering in fingerings:
        if midi_to_fingerings_dict.get(fingering.midi, None) is None:
            midi_to_fingerings_dict[fingering.midi] = [fingering]
        else:
            midi_to_fingerings_dict[fingering.midi].append(fingering)    

    fingering_predictor = fp.FingeringPrediction(model, midi_to_fingerings_dict)

    predicted_splits = []
    for i, split in enumerate(phrases): 
        midi_values = [note[0].pitch.midi for note in split]
        _, predictions = fingering_predictor.predict_fingerings(midi_values)
        predicted_split = []
        for i, (note, onset, offset) in enumerate(split):
            predicted_split.append((predictions[i], note, onset, offset))
        
        predicted_splits.append(predicted_split)

    phrases_difficulties = []
    for split in predicted_splits:
        phrases_difficulties.append(de.get_difficulties_of_sequence(split, model))

    return phrases_difficulties
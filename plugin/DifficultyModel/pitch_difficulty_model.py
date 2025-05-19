import music21
import numpy as np

def evaluate_difficulty(
        phrases: list[list[tuple[music21.note.Note, float, float]]], 
        additional_info: dict = dict()
        ):
    
    phrases_difficulties = []

    pitches = []
    for phrase in phrases:
        for note, _, _ in phrase:
            pitches.append(note.pitch.midi)

    min = np.min(pitches)
    max = np.max(pitches)

    for phrase in phrases:
        phrase_difficulties = []
        for note, onset, offset in phrase:
            phrase_difficulties.append(int((float(max - note.pitch.midi) / float(max - min)) * 100))

        phrases_difficulties.append(phrase_difficulties)

    return phrases_difficulties


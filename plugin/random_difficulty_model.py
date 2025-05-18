import music21
import random

def evaluate_difficulty(
        phrases: list[list[tuple[music21.note.Note, float, float]]]
        ):
    
    phrases_difficulties = []

    for phrase in phrases:
        phrase_difficulties = []
        for note, onset, offset in phrase:
            phrase_difficulties.append(random.randint(0, 100))

        phrases_difficulties.append(phrase_difficulties)

    return phrases_difficulties


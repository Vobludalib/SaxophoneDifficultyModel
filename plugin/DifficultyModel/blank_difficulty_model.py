import music21

"""
Outlined below is the bare interface that will be used by the model_to_visualisation script and is required for correct operation.

The script works with a class the exposes a method called evaluate_difficulty with parameters as defined below
"""


class BlankModel:
    def evaluate_difficulty(
            self,
            phrases: list[list[tuple[music21.note.Note, float, float]]], # Each element of the outer list is a musical phrase (inner list). Each inner list contains tuples of music21 Notes, the onset, and offset time respectively.
            ):
        
        phrases_difficulties = []

        for phrase in phrases:
            phrase_difficulties = []
            for note, onset, offset in phrase:
                phrase_difficulties.append(0) # Set the difficulty for the given note (0 to 100 --- anything below 0 will be clipped to 0; anything above 100 will be clipped to 100)

            phrases_difficulties.append(phrase_difficulties)

        return phrases_difficulties


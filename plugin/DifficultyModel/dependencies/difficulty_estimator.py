import dependencies.model as m

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

    curr_to_next_predicted_ts = m.predict_fingering_transition(difficulty_model, curr_fingering, next_fingering)

    curr_to_next_percent = (curr_to_next_estimated_ts / curr_to_next_predicted_ts) * 100

    difficulties.append(curr_to_next_percent)

    if len(sequence) == 2:
        previous_index = 0
        central_index = 1

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

        prev_to_curr_predicted_ts = m.predict_fingering_transition(difficulty_model, prev_fingering, curr_fingering)
        curr_to_next_predicted_ts = m.predict_fingering_transition(difficulty_model, curr_fingering, next_fingering)

        prev_to_curr_percent = (prev_to_curr_estimated_ts / prev_to_curr_predicted_ts) * 100
        curr_to_next_percent = (curr_to_next_estimated_ts / curr_to_next_predicted_ts) * 100

        if prev_to_curr_percent > 100 or curr_to_next_percent > 100:
            difficulties.append(max(prev_to_curr_percent, curr_to_next_percent))
        else:
            difficulties.append(prev_to_curr_percent * 0.5 + curr_to_next_percent * 0.5)

    # Last note:
    prev_onset = sequence[previous_index][2]
    prev_fingering = sequence[previous_index][0]
    curr_onset = sequence[central_index][2]
    curr_offset = sequence[central_index][3]
    curr_fingering = sequence[central_index][0]

    prev_to_curr_estimated_ts = (1 / (curr_offset - prev_onset))

    prev_to_curr_predicted_ts = m.predict_fingering_transition(difficulty_model, prev_fingering, curr_fingering)

    prev_to_curr_percent = (prev_to_curr_estimated_ts / prev_to_curr_predicted_ts) * 100

    difficulties.append(prev_to_curr_percent)

    return difficulties
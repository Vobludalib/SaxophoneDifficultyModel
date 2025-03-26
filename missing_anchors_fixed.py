import encoding
import csv
import random

sessions_file = '/Users/slibricky/Desktop/Thesis/thesis/modular/files/final/sessions.csv'
fixed_sessions_file = '/Users/slibricky/Desktop/Thesis/thesis/modular/files/final/sessions_fixed.csv'
anchors_file = '/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/anchor_intervals.txt'
encodings_file = '/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt'

anchor_encoding_pairs = []

encoding_to_fingering = {}
with open(encodings_file, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader, None)
    for row in reader:
        midi = int(row[0].strip())
        name = row[1].strip()
        enc = row[2].strip()
        fing = encoding.Fingering(midi, name, enc)
        encoding_to_fingering[enc] = fing

with open(anchors_file, 'r') as file:
    reader = csv.reader(file)
    for line in reader:
        encoding1 = line[0]
        encoding2 = line[1]
        anchor_encoding_pairs.append((encoding1, encoding2))

header = None
with open(sessions_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)
    lines = []
    session_index = 0
    first_index_of_session = 0
    last_index_of_session = 0
    i = 0
    not_seen_anchors = anchor_encoding_pairs.copy()
    for line in reader:
        if int(line[0]) != session_index:
            last_index_of_session = i
            for not_seen_anchor in not_seen_anchors:
                insert_index = random.randint(first_index_of_session + 1, last_index_of_session)
                fing1 = encoding_to_fingering[not_seen_anchor[0]]
                fing2 = encoding_to_fingering[not_seen_anchor[1]]
                lines.insert(insert_index, [session_index, -1, fing1.midi, fing1.name, fing1.generate_encoding(), fing2.midi, fing2.name, fing2.generate_encoding()])
            session_index = int(line[0])
            first_index_of_session = i
            not_seen_anchors = anchor_encoding_pairs.copy()
        cluster = int(line[1])
        if cluster == -1:
            encoding1 = line[4]
            encoding2 = line[7]
            not_seen_anchors.remove((encoding1, encoding2))
        lines.append(line)
        i += 1

with open(fixed_sessions_file, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(lines)
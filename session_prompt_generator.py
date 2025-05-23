from music21 import *
import music21
import argparse
import csv
import img2pdf
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
                    prog='Generate trills order for data collection')
    parser.add_argument('-i', '--inp', type=str)
    parser.add_argument('-o', '--out', type=str, help="Path to directory where output should be stored")
    args = parser.parse_args()

    temp_folder = os.path.join(".", "temp")
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    for filename in os.listdir(temp_folder):
        file_path = os.path.join(temp_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    sessions = {}
    with open(args.inp, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, line in tqdm(enumerate(reader)):
            session = int(line[0])
            cluster = int(line[1])
            note1midi = int(line[2])
            note1name = line[3]
            note2midi = int(line[5])
            note2name = line[6]
            if session not in sessions:
                sessions[session] = [[cluster, note1midi, note1name, note2midi, note2name]]
            else:
                sessions[session].append([cluster, note1midi, note1name, note2midi, note2name])

            note1 = note.Note(note1midi, name=note1name)
            note2 = note.Note(note2midi, name=note2name)
            note3 = note.Note(note1midi, name=note1name)
            tb1 = music21.expressions.TextExpression(note1name)
            tb2 = music21.expressions.TextExpression(note2name)
            ts = expressions.TremoloSpanner()
            ts.addSpannedElements([note1, note2])
            ts.numberOfMarks = 3
            if note1midi > note2midi: tb1.placement = 'above'
            else: tb2.placement = 'above'
            s = stream.Stream()
            s.insert(0, note1)
            s.insert(1, note2)
            s.insert(2, note3)
            s.insert(0, tb1)
            s.insert(1, tb2)
            s.insert(0, ts)
            s.insert(0, metadata.Metadata())
            s.metadata.title = f'Session {session}'
            s.metadata.composer = f'Trill #{i} - cluster {cluster}'
            s.write('musicxml.png', fp=f'temp/S{session}-{i}-{cluster}-{note1name.replace(' ', '').replace('#', 'sharp').replace('(', '').replace(')', '')}-{note2name.replace(' ', '').replace('#', 'sharp').replace('(', '').replace(')', '')}-{i}')

    for session in sessions:
        with open(f"{args.out}"f"session{session}.pdf", "wb") as f:
            list_of_i_bound_to_session = [i for i in os.listdir('temp') if (i.endswith(".png") and f"S{session}-" in i)]
            ints_of_i = [int(i.split('.')[0].split('-')[-2]) for i in list_of_i_bound_to_session]
            list_of_i_bound_to_session = [(list_of_i_bound_to_session[i], ints_of_i[i]) for i in range(len(ints_of_i))]
            list_of_i_bound_to_session.sort(key=lambda tup: tup[1])
            list_of_i_bound_to_session = [tup[0] for tup in list_of_i_bound_to_session]
            session_files = [open("./temp/" + str(i), 'rb') for i in list_of_i_bound_to_session]
            f.write(img2pdf.convert(session_files))

if __name__ == '__main__':
    main()
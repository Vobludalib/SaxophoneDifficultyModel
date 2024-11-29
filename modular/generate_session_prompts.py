from music21 import *
import music21
import encoding
import argparse
import csv

def main():
    parser = argparse.ArgumentParser(
                    prog='Generate trills order for data collection')
    parser.add_argument('--inp', type=str)
    parser.add_argument('--out', type=str, help="Where to store output", default="prompts.pdf")
    args = parser.parse_args()

    sessions = {}
    with open(args.inp, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for line in reader:
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
            tb1 = music21.expressions.TextExpression(note1name)
            tb2 = music21.expressions.TextExpression(note2name)
            if note1midi > note2midi: tb1.placement = 'above'
            else: tb2.placement = 'above'
            s = stream.Stream()
            s.insert(0, note1)
            s.insert(1, note2)
            s.insert(0, tb1)
            s.insert(1, tb2)
            s.write('musicxml.pdf', fp=f'temp/S{session}-{cluster}-{note1name}-{note2name}.pdf')

if __name__ == '__main__':
    main()
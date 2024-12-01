from music21 import *
import music21
import encoding
import argparse
import csv
import img2pdf
import os

def main():
    parser = argparse.ArgumentParser(
                    prog='Generate trills order for data collection')
    parser.add_argument('--inp', type=str)
    parser.add_argument('--out', type=str, help="Where to store output", default="prompts.pdf")
    args = parser.parse_args()
    args.inp = "/Users/slibricky/Desktop/Thesis/thesis/modular/files/InitialSessionsGenerated.csv"

    sessions = {}
    with open(args.inp, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, line in enumerate(reader):
            session = int(line[0])
            if session != 0: continue
            cluster = int(line[1])
            note1midi = int(line[2])
            note1name = line[3]
            note2midi = int(line[5])
            note2name = line[6]
            if session not in sessions:
                sessions[session] = [[cluster, note1midi, note1name, note2midi, note2name]]
            else:
                sessions[session].append([cluster, note1midi, note1name, note2midi, note2name])

            # note1 = note.Note(note1midi, name=note1name)
            # note2 = note.Note(note2midi, name=note2name)
            # note3 = note.Note(note1midi, name=note1name)
            # tb1 = music21.expressions.TextExpression(note1name)
            # tb2 = music21.expressions.TextExpression(note2name)
            # ts = expressions.TremoloSpanner()
            # ts.addSpannedElements([note1, note2])
            # ts.numberOfMarks = 3
            # if note1midi > note2midi: tb1.placement = 'above'
            # else: tb2.placement = 'above'
            # s = stream.Stream()
            # s.insert(0, note1)
            # s.insert(1, note2)
            # s.insert(2, note3)
            # s.insert(0, tb1)
            # s.insert(1, tb2)
            # s.insert(0, ts)
            # s.insert(0, metadata.Metadata())
            # s.metadata.title = f'Session {session}'
            # s.metadata.composer = f'Trill #{i} - cluster {cluster}'
            # s.write('musicxml.png', fp=f'temp/S{session}-{i}-{cluster}-{note1name.replace(' ', '').replace('#', 'sharp').replace('(', '').replace(')', '')}-{note2name.replace(' ', '').replace('#', 'sharp').replace('(', '').replace(')', '')}')

    for session in sessions:
        print(session)
        with open(f"session{session}.pdf", "wb") as f:
            session_files = [open("./temp/" + i, 'rb') for i in os.listdir('temp') if (i.endswith(".png") and f"S{session}-" in i)]
            print(session_files)
            f.write(img2pdf.convert(session_files))

if __name__ == '__main__':
    main()
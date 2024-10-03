import csv
import music21
from tqdm import tqdm

pathToCsv = "/Users/slibricky/Desktop/Thesis/melospy-gui_V_1_4b_mac_osx/bin/analysis/feature+viz/BigramsTS.csv"

TRANSPOSEINTOBB = True

bigramDict = { }

def parseBigramsIntoDict(d: dict, bigramString):
    bigrams = str.split(bigramString, ':')
    for bigram in bigrams:
        d[bigram] = d.get(bigram, 0) + 1

with open(pathToCsv, 'r') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=";")
    header = next(csvReader, None)
    for row in csvReader:
        parseBigramsIntoDict(bigramDict, row[1])

outputDict = dict(sorted(bigramDict.items(), key= lambda x: x[1]))

for key in outputDict:
    pitches = str.split(key, ',')
    pitch1 = music21.note.Note(int(pitches[0]) + 2 if TRANSPOSEINTOBB else 0).nameWithOctave
    pitch2 = music21.note.Note(int(pitches[1]) + 2 if TRANSPOSEINTOBB else 0).nameWithOctave
    print(f"Interval {pitch1} -> {pitch2} appears {bigramDict[key]} times")

for i in range(44, 76):
    for j in range (44, 76):
        val = bigramDict.get(f"{i},{j}", -1)
        if val == -1 or val == 0:
            pitch1 = music21.note.Note(int(i) + 2 if TRANSPOSEINTOBB else 0)
            pitch2 = music21.note.Note(int(j) + 2 if TRANSPOSEINTOBB else 0)
            print(f"Interval from {pitch1.nameWithOctave} to {pitch2.nameWithOctave} not found in Tenor solos")
            if abs(pitch1.pitch.midi - pitch2.pitch.midi) < 20:
                print("======= WARNING!!!!! ========")

#45 is lowest in dataset - THIS IS IN BB
#92 is highest in dataset
# excluding altissimo wackiness, let's do Bb2 up to F#5
# 46 -> 78 transposed, so we have to subtract 2
# 44 -> 76 is the reasonable range
# let's check that each interval appears at least once
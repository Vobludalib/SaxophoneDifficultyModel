import trill_detection_tool
from encoding_reader import Fingering
import argparse
import os
import csv

def clear_console():

    if os.name == 'nt':
        _ = os.system('cls')

    else:
        _ = os.system('clear')

def main():
    parser = argparse.ArgumentParser(
                    prog='Batch Trill Parsing Tool',
                    description='Tool designed to automate fingering to trill lineup from csv files')
    parser.add_argument('-f', type=str, help="Path to the .csv file")
    parser.add_argument('-o', type=str, help="Will store all the outputs into one csv file with this name")
    args = parser.parse_args()
    if (not os.path.isfile(args.f)):
        print("-f is not a valid file")

    fingerings = {}
    with open("/Users/slibricky/Desktop/Thesis/thesis/encodings.txt", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader, None)
        for row in reader:
            midi = int(row[0].strip())
            name = row[1].strip()
            encoding = row[2].strip()
            fing = Fingering(midi, name, encoding)
            if midi in fingerings:
                fingerings[midi].append(fing)
            else:
                fingerings[midi] = [fing]

    newCsv = []
    newCsv.append(["Filename", "Midi 1", "Name 1", "Encoding 1", "Midi 2", "Name 2", "Encoding 2", "Trill speed"])
    with open(args.f, 'r') as csvf:
        reader = csv.reader(csvf)
        next(reader, None)
        for row in reader:
            filename = row[0].strip()
            midi1 = int(row[1].strip()) + 12
            midi2 = int(row[2].strip()) + 12
            trill_speed = float(row[3].strip())
            print(trill_speed)
            midi1fingering = None
            midi2fingering = None

            midi1_fingerings = len(fingerings.get(midi1, []))
            if (midi1_fingerings == 0):
                print(f"{midi1} has no known fingerings encoded")
            if (midi1_fingerings == 1):
                midi1fingering = fingerings[midi1][0]
            if (midi1_fingerings > 1):
                clear_console()
                print(f"Given midi note {midi1} has multiple fingerings, choose which one was associated with:")
                print(f"File: {filename}")
                for i in range(1, midi1_fingerings + 1):
                    print(f"{i}. {fingerings[midi1][i-1].name} - {fingerings[midi1][i-1].generate_encoding()}")
                inp = int(input())
                midi1fingering = fingerings[midi1][inp - 1]
            
            midi2_fingerings = len(fingerings.get(midi2, []))
            if (midi2_fingerings == 0):
                print(f"{midi2} has no known fingerings encoded")
            if (midi2_fingerings == 1):
                midi2fingering = fingerings[midi2][0]
            if (midi2_fingerings > 1):
                clear_console()
                print(f"Given midi note {midi2} has multiple fingerings, choose which one was associated with:")
                print(f"File: {filename}")
                for i in range(1, midi2_fingerings + 1):
                    print(f"{i}. {fingerings[midi2][i-1].name} - {fingerings[midi2][i-1].generate_encoding()}")
                inp = int(input())
                midi2fingering = fingerings[midi2][inp - 1]

            append_row = [filename, midi1, midi1fingering.name, midi1fingering.generate_encoding(), midi2, midi2fingering.name, midi2fingering.generate_encoding(), trill_speed]
            newCsv.append(append_row)

    with open(args.o, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerows(newCsv)     
        
    
if __name__ == '__main__':
    main()
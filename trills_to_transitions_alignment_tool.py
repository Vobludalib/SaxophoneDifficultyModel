from encoding import Fingering
import argparse
import os
import csv

encodings_path = os.path.join(".", "encodings", "encodings.txt")

def clear_console():
    if os.name == 'nt':
        _ = os.system('cls')

    else:
        _ = os.system('clear')

def main():
    parser = argparse.ArgumentParser(
                    prog='Trill to Transition Alignment tool',
                    description='Tool designed to automate fingering to trill lineup from csv files')
    parser.add_argument('-t', '--trills', type=str, help="Path to the .csv file of the trill speed estimates")
    parser.add_argument('-o', '--out', type=str, help="Will store all the outputs into one csv file at this path")
    parser.add_argument('-s', '--session', type=str, help="Path to the .csv file of the sessions being analysed")
    parser.add_argument('-v', '--verbose', action="store_true", help="If present, will give options to override alternate fingerings")
    args = parser.parse_args()
    if (not os.path.isfile(args.trills)):
        print("--estimates is not a valid file")
    if (not os.path.isfile(args.session)):
        print("--session is not a valid file")

    # Alternate loading used here to load directly indexed by midi value
    fingerings = {}
    with open(encodings_path, "r") as csvfile:
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

    expected_order = []
    if (args.session is not None):
        with open(args.session, 'r') as session_csv:
            reader = csv.reader(session_csv)
            header = next(reader, None)
            if "session" in header[0].lower():
                clear_console()
                print(f"Session csv file contains \"Session\" in the header, so I expect multiple sessions to be in the same file!")
                print(f"Please enter the session ID to analyse:")
                expected_session_id = int(input())
            # Header is:
            # Session,Cluster,Note 1 midi,Note 1 name,Note 1 encoding,Note 2 midi,Note 2 encoding,Note 2 encoding
            for row in reader:
                session = int(row[0])
                if session != expected_session_id:
                    continue
                cluster = int(row[1])
                note_1_midi = int(row[2])
                note_1_name = row[3]
                note_1_encoding = row[4]
                note_2_midi = int(row[5])
                note_2_name = row[6]
                note_2_encoding = row[7]
                expected_order.append( {
                    "cluster": cluster, 
                    "note_1_midi": note_1_midi, 
                    "note_1_name": note_1_name,
                    "note_1_encoding": note_1_encoding,
                    "note_2_midi": note_2_midi,
                    "note_2_name": note_2_name,
                    "note_2_encoding": note_2_encoding
                    } )

    enforce_order = expected_order is not []
    trill_detected_csv = []
    with open(args.trills, 'r') as csvf:
        reader = csv.reader(csvf)
        next(reader, None)
        for row in reader:
            filename = row[0].strip()
            # We will sort by any numbers present in the file name
            # e.g. "Audio 15 (1)" will become 151 - BE CAREFUL IN HOW FILE NAMES EXPORT, AS THEY ENFORCE ORDERING
            extract_digits = "".join([d for d in filename if d.isdigit()])
            extracted_digits = int(extract_digits)
            #TODO: Change +12 to +14, remove +2 in trill_detection tool
            midi1 = int(row[1].strip()) + 12
            midi2 = int(row[2].strip()) + 12
            trill_speed = float(row[3].strip())
            midi1fingering = None
            midi2fingering = None
            if midi1 > midi2:
                midi1, midi2 = midi2, midi1
            append_row = [filename, extracted_digits, midi1, midi2, trill_speed]
            trill_detected_csv.append(append_row)

    # Sort newCSV by extracted_digits
    trill_detected_csv.sort( key=lambda row: row[1] )

    if (len(expected_order) != len(trill_detected_csv)):
        print(f"Expected {len(expected_order)} as per the session csv, but found {len(trill_detected_csv)} from the trill detected csv.")
        input()
        exit()

    # try:
    output = []
    for i, row in enumerate(trill_detected_csv):
        expected_midi_notes = [expected_order[i]['note_1_midi'], expected_order[i]['note_2_midi']]
        clear_console()
        print(f"Processing audio file: {trill_detected_csv[i][0]}")
        print(f"Expected: {expected_midi_notes[0]}, {expected_midi_notes[1]}")
        print(f"Detected: {row[2]}, {row[3]} at speed {row[4]}")
        input()

        # Check MIDI values correspond
        if (row[2] not in expected_midi_notes or row[3] not in expected_midi_notes):
            print(f"Mismatch in expected and detected notes")
            print(f"If you want to override the detected frequencies with the expected ones, enter 'Y'")
            print(f"NOTE! Only do this if you are certain that the detected frequencies are meant to be overriden (i.e. bad detection/tuning)")
            answer = input()
            if answer == 'Y' or answer == "":
                midi1 = expected_midi_notes[0]
                midi2 = expected_midi_notes[1]
            else:
                midi1 = row[2]
                midi2 = row[3]
        else:
            midi1 = row[2]
            midi2 = row[3]

        midi1_fingerings = len(fingerings.get(midi1, []))
        if (midi1_fingerings == 0):
            print(f"{trill_detected_csv[i][0]}")
            print(f"{midi1} has no known fingerings encoded")
            input()
        if (midi1_fingerings == 1):
            midi1fingering = fingerings[midi1][0]
        if (midi1_fingerings > 1):
            inp = ""
            expected_fingering1 = [fingering for fingering in fingerings[midi1] if expected_order[i]['note_1_encoding'] == fingering.generate_encoding()][0]
            if args.verbose:
                clear_console()
                print(f"Given midi note {midi1} has multiple fingerings, choose which one was associated with:")
                print(f"File: {trill_detected_csv[i][0]}")
                print(f"Expected from session csv:")
                print(f"1. {expected_fingering1.name} - {expected_fingering1.generate_encoding()}")
                print(f"Manual override")
                for j in range(2, 2 + midi1_fingerings):
                    print(f"{j}. {fingerings[midi1][j-2].name} - {fingerings[midi1][j-2].generate_encoding()}")
                
                inp = input()
            if inp == "" or int(inp) == 1:
                midi1fingering = expected_fingering1
            else:
                midi1fingering = fingerings[midi1][int(inp) - 2]
        
        # TODO: Get rid of code duplication
        midi2_fingerings = len(fingerings.get(midi2, []))
        if (midi2_fingerings == 0):
            print(f"{trill_detected_csv[i][0]}")
            print(f"{midi2} has no known fingerings encoded")
            input()
        if (midi2_fingerings == 1):
            midi2fingering = fingerings[midi2][0]
        if (midi2_fingerings > 1):
            inp = ""
            expected_fingering2 = [fingering for fingering in fingerings[midi2] if expected_order[i]['note_2_encoding'] == fingering.generate_encoding()][0]
            if args.verbose:
                clear_console()
                print(f"Given midi note {midi2} has multiple fingerings, choose which one was associated with:")
                print(f"File: {trill_detected_csv[i][0]}")
                print(f"Expected from session csv:")
                print(f"{midi1}, {midi2}")
                print(f"1. {expected_fingering2.name} - {expected_fingering2.generate_encoding()}")
                print(f"Manual override")
                for j in range(2, 2 + midi2_fingerings):
                    print(f"{j}. {fingerings[midi2][j-2].name} - {fingerings[midi2][j-2].generate_encoding()}")
                inp = input()
            if inp == "" or int(inp) == 1:
                midi2fingering = expected_fingering2
            else:
                midi2fingering = fingerings[midi2][int(inp) - 2]

        output.append([trill_detected_csv[i][0],expected_order[i]['cluster'],midi1,midi1fingering.name,midi1fingering.generate_encoding(),midi2,midi2fingering.name,midi2fingering.generate_encoding(), trill_detected_csv[i][4]])

    with open(args.o, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['Filename', 'Cluster', 'Midi 1 (transposed as written for TS)', 'Fingering 1 Name', 'Fingering 1 Encoding', 'Midi 2 (transposed as written for TS)', 'Fingering 2 Name', 'Fingering 2 Encoding', 'Trill Speed'])
        writer.writerows(output)
    
if __name__ == '__main__':
    main()
import os
import string
import csv

dir_path = "/Users/slibricky/Desktop/Thesis/raw_data/"

def extract_indices(input_string):
    extracted_indices = ''.join(list(filter(lambda s: s in string.digits, input_string)))
    session_index = int(extracted_indices)
    return session_index

subfolders = [ f.path for f in os.scandir(dir_path) if f.is_dir() ]

subfolder_to_index = {}
for folder in subfolders:
    subfolder_to_index[folder] = extract_indices(folder)

subfolders.sort(key=lambda x: subfolder_to_index[x])
session_to_player = [folder.split('/')[-1].split('-')[0] for folder in subfolders]

old_to_new_mapping = {}
old_to_new_files_mapping = {}
repeated_old_monitoring = {}
repeated_old_names = []
new_name_to_player = {}
new_name_to_session = {}
i = 0
for session_i, folder in enumerate(subfolders):
    file_order = []
    file_to_index = {}
    for dirpath, _, files in os.walk(folder):
        for file in files:
            file, extension = os.path.splitext(file)
            if extension == ".wav":
                file_order.append(file)
                index = extract_indices(file)
                file_to_index[file] = index
            else:
                os.remove(os.path.join(folder, f"{file}{extension}"))
    
    file_order.sort(key=lambda x: file_to_index[x])

    for file in file_order:
        if repeated_old_monitoring.get(file, None) is None:
            repeated_old_monitoring[file] = [folder]
        else:
            repeated_old_monitoring[file].append(folder)

        # old_path = os.path.join(folder, f"{file}.wav")
        # new_path = os.path.join(folder, f"Trill{i}_renamed.wav")
        # print(f"{old_path}: {new_path}")
        # os.rename(old_path, new_path)
        # i += 1

        if "renamed" in file:
            new_name = file.split('_')[0]
            print(f"{file} -> {new_name}")
            old_path = os.path.join(folder, f"{file}.wav")
            new_path = os.path.join(folder, f"{new_name}.wav")
            os.rename(old_path, new_path)



dir_path = "/Users/slibricky/Desktop/Thesis/thesis/modular/files/data_processed/"

files = [ f.path for f in os.scandir(dir_path) if f.is_file() ]

file_to_index = {}
for file in files:
    file_to_index[file] = extract_indices(file)

files.sort(key=lambda x: file_to_index[x])
session_to_player = [file.split('/')[-1].split('Session')[0] for file in files]

new_rows = []
new_rows.append(["Filename", "Cluster", "Player", "Session", "Midi 1 (transposed as written for TS)", "Fingering 1 Name", "Fingering 1 Encoding", "Midi 2 (transposed as written for TS)", "Fingering 2 Name", "Fingering 2 Encoding", "Trill Speed"])
i = 0
for session_i, file in enumerate(files):
    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            row[0] = f"Trill{i}.wav"
            i += 1
            row.insert(2, session_to_player[session_i])
            row.insert(3, session_i)
            new_rows.append(row)

with open('./test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(new_rows)
import csv
import encoding

fingerings = encoding.load_fingerings_from_file('/Users/slibricky/Desktop/Thesis/thesis/modular/documentation/encodings.txt')
name_to_encoding_dict = {}
for fingering in fingerings:
    name_to_encoding_dict[fingering.name] = fingering.generate_encoding()

newCsv = []
header = None
with open('/Users/slibricky/Desktop/Thesis/thesis/modular/files/final/sessions_fixed.csv') as file:
    reader = csv.reader(file)
    header = next(reader, None)
    i = 2
    for row in reader:
        name1 = row[3]
        encoding1 = row[4]
        name2 = row[6]
        encoding2 = row[7]
        if name_to_encoding_dict[name1] != encoding1:
            print(f"line {i} -> FIRST encoding wrong")
            row[4] = name_to_encoding_dict[name1]
        if name_to_encoding_dict[name2] != encoding2:
            print(f"line {i} -> SECOND encoding wrong")
            row[7] = name_to_encoding_dict[name2]

        newCsv.append(row)

        i += 1

with open('./sessions_fixed.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(newCsv)

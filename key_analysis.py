import pprint
import sqlite3
import matplotlib.pyplot as plt
import music21

dbfile = r"/Users/slibricky/Desktop/Thesis/wjazzd.db"
con = sqlite3.connect(dbfile)

cursor = con.cursor()

keys = [a for a in cursor.execute('SELECT key FROM solo_info WHERE instrument="ts" OR instrument="cl" OR instrument="tp" OR instrument="cor"')]
keys_dict = {}
for key in keys:
    keys_dict[key] = keys_dict.get(key, 0) + 1

tonalityDegrees = {}
for key in keys_dict.keys():
    keyStr = str.split(key[0], "-")
    if (len(keyStr) != 2):
        continue
    if (keyStr[1] == "maj"):
        m21key = music21.key.Key(keyStr[0])
    elif (keyStr[1] == "min"):
        m21key = music21.key.Key(keyStr[0], "minor")
    else:
        print(f"Not considering {key}")
        continue

    tonalityDegree = m21key.sharps
    print(tonalityDegree)
    tonalityDegrees[tonalityDegree] = tonalityDegrees.get(tonalityDegree, 0) + keys_dict[key]

tonalities = [i for i in range(-12, 13)]
occurences = [tonalityDegrees.get(i, 0) for i in range(-12, 13)]
plt.bar(tonalities, occurences)
plt.show()
pprint.pprint(tonalityDegrees)
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sklearn.cluster as skc
import scipy.signal
from scipy.signal.windows import gaussian

print("Loading audio!")
audio, sample_rate = librosa.load(r"/Users/slibricky/Desktop/Thesis/thesis/testTrill.wav")
print("Finished loading!")
f_0, _, _ = librosa.pyin(y=audio, fmin=librosa.note_to_hz("A2"), fmax=librosa.note_to_hz("C5"), frame_length=1024)
sample_length = 22050/1024
midi_vals = np.asarray(librosa.hz_to_midi(list(f_0)))
times = [i*46.5 for i in range (0, len(midi_vals))]
midi_vals = midi_vals[1:]
two_means_cluster = skc.KMeans(n_clusters=2).fit(midi_vals.reshape(-1, 1))
notes1 = []
notes2 = []
for i in range(0, midi_vals.shape[0]):
    if two_means_cluster.labels_[i] == 0:
        notes1.append(midi_vals[i])
    else:
        notes2.append(midi_vals[i])
notes1 = np.asarray(notes1)
notes2 = np.asarray(notes2)
note1_midi = round(np.median(notes1))
note2_midi = round(np.median(notes2))

changes = 0
last_midi_val = 0
time_spent_on_note = 0
other_midi_val = 0
for midi_val in midi_vals:
    if last_midi_val == 0:
        if round(midi_val) == note1_midi:
            last_midi_val = note1_midi
            other_midi_val = note2_midi
            continue
        elif round(midi_val) == note2_midi:
            last_midi_val = note2_midi
            other_midi_val = note1_midi
            continue
    
    if round(midi_val) == last_midi_val:
        time_spent_on_note += 1
        continue   

    if round(midi_val) == other_midi_val and time_spent_on_note > 2:
        changes += 1
        last_midi_val, other_midi_val = other_midi_val, last_midi_val
        time_spent_on_note = 0

print(changes)

# plt.plot(times[1:], midi_vals)
# plt.savefig(r"./trill.png")
## Data collection considerations:

### [Workflow diagram](https://drive.google.com/file/d/1tdtRzK2rv-4AenHuXRCl5gKNM3K1JlUI/view?usp=sharing)

### Prompt:
<ul>
<li>
Tell participants to focus on speed, clarity, 'cleanliness' and tuning of every transition.
</li>
<li>
Print a notation of the two notes to trill between - for notes with multiple fingerings add text describing fingering via text - doable via m21.
</li>
<li>
All notes will be presented in transposed pitch (i.e. the way the musicians are used to reading them for given instrument).
</li>
<li>
Present some kind of tremolo, and show what note to end on (so as to prevent half-trills)
</li>
</ul>

### Processing workflow:
#### This is now outdated somewhat
<ol>
<li>
Record simultaneous audio + video (can be seperate or not - sync via clap if needed).
</li>
<li>
Manually trim trills from other audio (inspecting waveforms in DAW) - at the same time, trill the synced video.
</li>
<li>
Export every trill on its own with naming of files in order they appear in the original audio.
</li>
<li>
Use automatic trill detection tool (see trill_detection_tool.py) to detect the notes and speed of each trill - should export to csv or some other intermediate format that allows semi-filled information.
</li>
<li>
For every trill try to assign the fingerings used - if a note has only one fingering it is assigned. If a note has multiple fingerings, then a temporary value warning the user is put there instead.
</li>
<li>
For every trill that has this temporary value, go to the relevant video and figure out which fingerings were used - this will be presented in the form of a 'click on the correct fingering' dialogue, to prevent user entry error.
</li>
</ol>

### Algorithm for trill speed + note detection
When presented with an audio file of a trill, we first let Crepe do f0 detection.

Given we have the matrix of f0 predictions and voicing likelihoods, we filter out only frames above a given voicing threshold (to get only frames where notes were being played). 

Now, we perform k-means clustering on the detecting voiced frames (i.e. cluster by their frequency). First, we cluster with k=2. This works well in case there are two very cleanly seperated notes with no in betweens.  However, in case of some more difficult trills, we find that (on saxophone) there is a possibility of f0 detection showing under or overblown (i.e. up or down an octave) notes. These frequencies do occur, but they are not really of interesting to us. As such, if we find that the intraclass compactness of a given cluster is above an allowed limit (i.e. more than one note is in the cluster), then we can assume that our clustering didn't isolate two notes cleanly. 

In this case, we try clustering at k=3 and k=4, and then we try to merge two clustes if they correspond to octaves shifts (e.g. k1 = D3, k2 = D4, k3 = G3). In these cases, then we present the choice to the user to choose what the actual notes are. And perform trill detection on the merged clusters.

The actual trill speed counting occurs on a given interval (more on this later) by seeing how many times we go from cluster 1 to cluster 2 and back, staying at each note for a given minimum amount of frames (to prevent flukes).

For each audio, we split it into a set amount of intervals, as there is a trend to speed up or slow down over the trill, due to the process of finding the speed limit / getting tired. For example, if we want 3 intervals, we split the audio frames into 0-50%, 25-75%, 50-100%, and then choose the fastest interval.

Possible additions to this would be prevent full filtering out of unvoiced frames, instead opting to count them as a way of detecting how legato the trill is. This is actually a great way of adding another dimension to difficulty, as trill speed is not the be-all and end-all.
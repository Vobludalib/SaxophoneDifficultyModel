# Motivation

The eventual goal is to create a fingering automaton for arbitrary musical instruments (those that have set fingerings for given notes) using data-driven methodology, so that machine-learning and statistics models can be used.

The main problem with data-driven approaches is the data acquisition. When trying to get a recording of every possible fingering pair we encounter the issue of non-linear blow-up - adding one new fingering requires us to get data for it to every other fingering that is present.

To combat this, we want to find a better way to incorporate data into the edge weights of the automaton that does not require as much data to be recorded.

### What is the data?

A model of fingering transition difficulty was chosen that is based on the trill (or tremolo) speed of the given fingering pair. We instruct the performers to play the given trill/tremolo as fast (and a few others things, see later) as possible, to get the upper bound of the speed, as this is assumed to be a good-enough representation of both the difficutly with respect to both the motorics and also the actual nature of sound production.

The raw data will be a series of audio recordings of performers performing these trills on their given instruments. Processing this data will lend us a mapping between fingering pairs and trill speeds. The technical details of this are described later.

### How do we get this data?

The principle idea is that we can figure out which fingering transitions are similar (i.e. they are similarly difficult), and then use that knowledge to infer the difficulty of fingerings we do not have data for.

As such, when recording, we choose a representative sample of the instrument fingerings to record that will get us maximal coverage.

### Why do we need similarity?

When we go to record the data, to ensure proper coverage of all the possible difficulties of fingering transitions, we need to try to get a representative sample of fingerings that will give us the 'most information' about the fingering transitions we do not record.

We can get some rough idea of the frequency of fingering bigrams by analysing the Weimer Jazz dataset for a given instrument (assuming it has enough entries). When we look at the outputs, we see that some very simple fingering transitions appear much more frequently than some more obscure ones, but sampling based on this distribution is ill-fated, as then we will oversample fingerings that are similar to each other and could be predicted based on a recording of only a few of them.

At the end of the day, we want this automaton to be representative of the extremes of the instrument, meaning that the more obsure fingerings are actually the ones that we might pay more attention to, as these are the ones that will have the biggest differences between each other in their difficulty.

However, this has to be weighted with the fact that the more common fingerings are also the most likely input to the automaton, and so it would be a shame to give fully equal weight to the extremes of the instrument, as for most cases we would be learning things that will never be utilised.

Also, when considering *ALL* combinations of fingerings, the vast majority of them basically do not appear (or very infrequently), as most of them will become jumps across the range of the instrument that are very rare in actual music that is played, as they do not lend itself to the stylistic considerations of that music.

As such, an approach was/is chosen to work on this similarity-based sampling that gets us a representative sample, with some doubling up of more common transitions to get better data for the most-represented input scenarios.

### What does similar mean?

When discussing 'similar' fingerings, we cannot lean on automatic feature extraction from the fingering encodings, as we do not have any of the data required to do this. 

As such, similar here refers to similarity between features extracted from the given fingering pair encodings. These features to be extracted are domain specific and expertly defined. Usually this will be an amalgamation of features that performers consider influential when deciding how hard a given tremolo is and pedagogical literature that might identify problematic spots of fingering on the instrument.

### What to do with this data? - research part

As part of the research, we will record every fingering pair on tenor saxophone at least once (with doubling up based on frequency in Weimar).

The initial research questions are:
- How good is a given clustering method at generating the similarity (soft equivalence) classes for fingerings? (test using a fixed feature extraction method)
- Is it possible to automate feature extraction given a tiny amount of recorded data?
- How do we best design the features to be extracted?
- What method is best at generating the expected (inferred) trill speed given features? Linear model / decision trees / etc.

# Data collection considerations:

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

<!-- Now, we perform k-means clustering on the detecting voiced frames (i.e. cluster by their frequency). First, we cluster with k=2. This works well in case there are two very cleanly seperated notes with no in betweens.  However, in case of some more difficult trills, we find that (on saxophone) there is a possibility of f0 detection showing under or overblown (i.e. up or down an octave) notes. These frequencies do occur, but they are not really of interesting to us. As such, if we find that the intraclass compactness of a given cluster is above an allowed limit (i.e. more than one note is in the cluster), then we can assume that our clustering didn't isolate two notes cleanly. 

In this case, we try clustering at k=3 and k=4, and then we try to merge two clustes if they correspond to octaves shifts (e.g. k1 = D3, k2 = D4, k3 = G3). In these cases, then we present the choice to the user to choose what the actual notes are. And perform trill detection on the merged clusters. -->

The actual trill speed counting occurs on a given interval (more on this later) by seeing how many times we go from cluster 1 to cluster 2 and back, staying at each note for a given minimum amount of frames (to prevent flukes).

For each audio, we split it into a set amount of intervals, as there is a trend to speed up or slow down over the trill, due to the process of finding the speed limit / getting tired. For example, if we want 3 intervals, we split the audio frames into 0-50%, 25-75%, 50-100%, and then choose the fastest interval.

Possible additions to this would be prevent full filtering out of unvoiced frames, instead opting to count them as a way of detecting how legato the trill is. This is actually a great way of adding another dimension to difficulty, as trill speed is not the be-all and end-all.
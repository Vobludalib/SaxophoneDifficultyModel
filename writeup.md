## Data collection considerations:

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
</ul>

### Processing:
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
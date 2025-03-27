## **How to recreate figures:**

#### Installing requirements
TODO: Create requirements.txt
pip install requirements.txt

#### encoding.py
encoding.py is used as a library throughout the other Python scripts, as it unifies operations of transitions, encodings, and feature extraction methods using an object-based approach.

#### Disclaimer for the future

All file paths are provided in the Unix/Mac format with '/' as the seperating character. You may have to change this on Windows.

#### Generating the sessions
The first step in recording the data would be to split all the transitions into chunks that will corresponding to what will be recorded in a given session by a given player. 

To generate the session .csv file that will be needed later, we can type the following into the terminal.

`python session_order_generator.py --noi 65 -o './test_sessions.csv'`

To see all possible command line options, you can do:

`python session_order_generator.py --help`

The specific sessions.csv file that corresponds to the data collected is in `./files/sessions/sessions.csv`.

NOTE: This sessions file was generated on with an older version of this script, and several adjustments had to be made on the day of recording sessions, so there is no way to get the code to generate this sessions csv file exactly.

For all future examples, when working with the data provided, use this specific sessions csv file.

#### Prompt generation
To produce the session prompts (as seen in Figure 1), we will use the session_prompt_generator.py script.

To generate the session prompts, we can use type the following on to the command line:

`python session_prompt_generator.py -i './files/sessions/sessions.csv' -o './files/sessions/prompts'`

As always, the output directory is up to you. This will work with the existing directory structure in the repository.

#### Trill detection procedure
The process of getting a .csv file with transition information and the actual trill speed works in two steps.
This workflow works at the level of sessions, so with the directory structure for the data I have chosen, you will need to run this workflow on every single session seperately.

1) First, you perform trill speed extraction plus prediction of the two notes that are being trilled between. This is done using trill_detection_tool.py
To perform this step, you can type the following on to the command line:
`python trill_detection_tool.py -f /path/to/submission_data/CHOOSE-A-SESSION/ -r -o /path/for/output.csv`
Substitute the example paths for the paths where you stored the raw audio data and where you want the output file to be stored.
This will perform the trill speed extraction and make a preliminary estimate of what the two notes being trilled between are - at this step, we do not yet know what given fingering is being for the notes.
2) We use a seperate script to perform alignment between the output from step 1 and our sessions .csv file that tells us what the fingerings used should have been.
To perform this step, type the following on to the command line:
`python trills_to_transitions_alignment_tool.py -t ./files/trills/CHOOSEASESSION.csv -s ./files/sessions/sessions.csv -o /path/to/output.csv`
When launching the script, you will be asked to choose a session, as the sessions.csv file contains multiple sessions. For example, if you have chosen to do alignment for DorianGraySession0Trills.csv, then you would enter '0' and press Enter at this point.
The tool will guide you through manual verification of the alignment (do the detected notes match the expected notes). If they do, then you can just press Enter and the alignment will occur automatically. 
In case the detected and expected notes do not match (due to tuning, bad detection, over/underblowing), then a different message will pop up alerting you to this. In this case, you open the .wav file in question, listen to the recording and get a manual ballpark estimate for the trill speed in trill/s (by finding out an appropriate BPM or something similar and doing the maths). If this speed roughly matches up with the trill speed in the prompt, you can press Enter and the detected notes will be overriden by the expected notes.
In case they do not match, there was either an issue with step 1, or you might have mislabeled your .wav files. Either way, this is for you to investigate. 
All the recordings as provided do not have problems with trill speed detection in step 1. However, some of them will get this misalignment prompts due to over/underblowing - all of these have been thoroughly checked, and they do not pose a problem.

#### Recreating figure 3
Figure 3 is a slightly cropped CREPE salience plot of one of the raw audio files. You can get a similar image for any of the raw audio files by doing:
`crepe /path/to/file.wav --save-plot`
This will return the salience plot in .png format.

#### Model tests
__From here on, I will be using the data in the attached data from the file called Processed_Data.csv__

<sub><sup>This .csv file was created by concatenating all the .csv files from trill extraction and session alignment as above. The Player and Session columns were added automatically after the fact to allow for the data to stand on its own. Normally, this information is just captured via the directory structure and directory naming scheme as can be seen in the data directory.<sup><sub>

To run the tests comparing the individual model and feature extractor combinations, we use model.py.

To run the model tests, we can do the following in the command line:
`python model.py -o /path/where/to/save/ -d /path/to/submission_data/Processed_Data.csv`

Default seeding for all the methods is what was used and can be seen in the output .csv files that report each model's performance.

Each model and feature extractor also generates the graph that was used for Figure 6.

#### Sampling tests
To generate the results of the sampling tests, we use sampling.py. 
We have to launch this script seperately for each sampling method we wish to test, as the runtime of each instance can be over 30 minutes.

To run the test (for example) for uniform sampling, we enter the following on to the command line:
`python sampling.py -d /path/to/data/Processed_Data.csv -o /path/to/output/directory --sampling_method uniform`

We can select 'uniform', 'cluster', and 'empirical' for the sampling_method argument.

Each instance creates a .csv file and .png file that show a summary of that specific sampling test.

To generate Figure 7, we have to use a standalone visualisation script called sampling_visualisation_script.py to combine the sampling test results.

To use this script, we can call it in the command line:
`python sampling_visualisation_script.py -s "METHOD" -p /path/to/that/samplingmethod/test.csv {repeated for each sampling method}`

<sub>It might look something like this:
`python sampling_visualisation_script.py --paths ./files/sampling_tests/full_data/test_896811.csv -s uniform -p ./files/sampling_tests/full_data/test_3697990.csv -s cluster -o ./files/sampling_tests -p ./files/sampling_tests/full_data/test_8375147.csv -s empirical`<sub>

#### Difficulty model
The difficulty model code is contained within difficulty_estimator.py (using fingering prediction code from fingering_prediction.py). To use this difficulty estimation using a model trained from the data we have, we can type the following on to the command line:

`python difficulty_estimator.py -i ./files/Short_Segment-Tenor_Saxophone.mxl -o ./files/Short_Segment_Annotated.mxl -d /path/to/data/Processed_Data.csv -e ./encodings/encodings.txt --bpm 160`

The code uses the E-HB (No EW) feature set.

#### Generating Figure 2

To generate the Figure 2 visualisation, we have the anchor_trills_visualiser.py tool. 

To use it, type the following on the command line:
`python anchor_trills_visualiser.py -d /path/to/data/Processed_Data.csv`

You can optionally add a `-o` parameter to dictate the output path in the form of a .png file.
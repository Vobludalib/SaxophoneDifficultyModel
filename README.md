# Saxophone difficulty modelling
By: Simon Libricky - [simonlibricky.com](www.simonlibricky.com)

As part of Bachelor's thesis at Charles University in Prague

#### Installing requirements
To install all the required Python libraries to run the code, you can run the following on the command line:
`pip install -r requirements.txt`

Make sure to navigate into the correct directory before running any of the command line commands here and below.

#### Data availability
The corpus of trill recordings is available attached in the submission .zip file. The files are placed into directories that correspond to the given performer and session in a unified naming format. ```Processed_Data.csv``` is the .csv file that is created by concatenating (and adding performer/session information) the extracted trill speeds for all the sessions. 

#### encoding.py
encoding.py is used as a library throughout the other Python scripts, as it unifies operations of transitions, encodings, and feature extraction methods using an object-based approach.

#### Disclaimer for the future

All file paths are provided in the Unix/Mac format with '/' as the seperating character. You may have to change this on Windows.

## Chapter 3

#### Generating the sessions (3.1.1)
The first step in recording the data would be to split all the transitions into chunks that will corresponding to what will be recorded in a given session by a given player. 

To generate the session .csv file that will be needed later, we can type the following into the terminal.

`python session_order_generator.py --noi 65 -o './test_sessions.csv'`

To see all possible command line options, you can do:

`python session_order_generator.py --help`

The specific sessions.csv file that corresponds to the data collected is in `./files/sessions/sessions.csv`.

NOTE: This sessions file was generated on with an older version of this script, and several adjustments had to be made on the day of recording sessions, so there is no way to get the code to generate this sessions csv file exactly.

For all future examples, when working with the data provided, use this specific sessions csv file.

#### Prompt generation (3.1.2)
To produce the session prompts (as seen in Figure 3.1), we will use the session_prompt_generator.py script.

To generate the session prompts, we can use type the following on to the command line:

`python session_prompt_generator.py -i './files/sessions/sessions.csv' -o './files/sessions/prompts'`

As always, the output directory is up to you. This will work with the existing directory structure in the repository.

#### Trill detection procedure (3.2)
The process of getting a .csv file with transition information and the actual trill speed works in two steps.
This workflow works at the level of sessions, so with the directory structure for the data I have chosen, you will need to run this workflow on every single session seperately.

This process takes a sizable chunk of time, so I do not recommend trying to perform this on more than one session at most, as it can take 30+ mins per session. From Chapter 5 onwards, I recommend using the provided ```Processed_Data.csv```.

1) **Section (3.2.1 - 3.2.3)**
First, you perform trill speed extraction plus prediction of the two notes that are being trilled between. This is done using trill_detection_tool.py
To perform this step, you can type the following on to the command line:
`python trill_detection_tool.py -f /path/to/submission_data/CHOOSE-A-SESSION/ -r -o /path/for/output.csv`
Substitute the example paths for the paths where you stored the raw audio data and where you want the output file to be stored.
This will perform the trill speed extraction and make a preliminary estimate of what the two notes being trilled between are - at this step, we do not yet know what given fingering is being for the notes.
2) **Section (3.2.4)**
We use a seperate script to perform alignment between the output from step 1 and our sessions .csv file that tells us what the fingerings used should have been.
To perform this step, type the following on to the command line:
`python trills_to_transitions_alignment_tool.py -t ./files/trills/CHOOSEASESSION.csv -s ./files/sessions/sessions.csv -o /path/to/output.csv`
When launching the script, you will be asked to choose a session, as the sessions.csv file contains multiple sessions. For example, if you have chosen to do alignment for DorianGraySession0Trills.csv, then you would enter '0' and press Enter at this point.
The tool will guide you through manual verification of the alignment (do the detected notes match the expected notes). If they do, then you can just press Enter and the alignment will occur automatically. 
In case the detected and expected notes do not match (due to tuning, bad detection, over/underblowing), then a different message will pop up alerting you to this. In this case, you open the .wav file in question, listen to the recording and get a manual ballpark estimate for the trill speed in trill/s (by finding out an appropriate BPM or something similar and doing the maths). If this speed roughly matches up with the trill speed in the prompt, you can press Enter and the detected notes will be overriden by the expected notes.
In case they do not match, there was either an issue with step 1, or you might have mislabeled your .wav files. Either way, this is for you to investigate. 
All the recordings as provided do not have problems with trill speed detection in step 1. However, some of them will get this misalignment prompts due to over/underblowing - all of these have been thoroughly checked, and they do not pose a problem.

#### Recreating Figure 3.3, 3.4, and 3.5
Figure 3.3 is a slightly cropped CREPE salience plot of one of the raw audio files. You can get a similar image for any of the raw audio files by doing:
`crepe /path/to/file.wav --save-plot`
This will return the salience plot in .png format.
Figure 3.4 and 3.5 are edited versions of different salience plots.

#### Recreating Figure 3.6 and 3.7
These figures are simple screenshots of the terminal when running the ```trills_to_transitions_alignment_tool.py```.

## Chapter 4
All the code related to Chapter 4 are present in ```encoding.py```. For clarification as to the specific way feature extraction is performed, please refer to the code. All of this is handled in a object-oriented way.

## Chapter 5

From here on, I will be using the data in the attached data from the file called Processed_Data.csv

<sub><sup>This .csv file was created by concatenating all the .csv files from trill extraction and session alignment as above. The Player and Session columns were added automatically after the fact to allow for the data to stand on its own. Normally, this information is just captured via the directory structure and directory naming scheme as can be seen in the data directory.<sup><sub>

#### Model tests (5.1.1 - 5.1.3)

To run the tests comparing the individual model and feature extractor combinations, we use model.py. Leaving the seed parameter alone should generate the same seeding we used.

To run the model tests, we can do the following in the command line:
`python model.py -o /path/where/to/save/ -d /path/to/submission_data/Processed_Data.csv -m {lm|mlp|lm-log|mlp-log}`

Default seeding for all the methods is what was used and can be seen in the output .csv files that report each model's performance.

Each model and feature extractor also generates the graph that was used for Figure 5.1.

For reference, here is the mapping from feature set to Python class
- R = RawFeatureExtractor
- F(PAF) = FingerFeatureExtractor(map_palm_to_fingers=False)
- F(P2FM) = FingerFeatureExtractor(map_palm_to_fingers=True)
- E-FB = ExpertFeatureIndividualFingersExtractor
- E-HB = ExpertFeatureNumberOfFingersExtractor

NoEW and NoM flags for Expert features are handled by parameters in the initializers of these classes.

#### Table 5.1, 5.2, and Figure 5.1
These tables and figures are generated by launching model.py with appropriate parameters.

#### Log space experiments (5.1.4)
These experiments are also handled by model.py, and by setting the --model flag to a '-log' variety.

#### Table 5.3
This table is generated by launching model.py with appropriate parameters.

## Chapter 6

#### Section 6.1-6.2
To generate the results of the sampling tests, we use sampling.py. 
We have to launch this script seperately for each sampling method we wish to test, as the runtime of each instance can be over 30 minutes. Again, the default seed parameter is the one we used.

To run the test (for example) for uniform sampling, we enter the following on to the command line:
`python sampling.py -d /path/to/data/Processed_Data.csv -o /path/to/output/directory --sampling_method uniform`

We can select 'uniform', 'cluster', and 'empirical' for the sampling_method argument.

Each instance creates a .csv file and .png file that show a summary of that specific sampling test.

#### Figure 6.1, and 6.2
To generate Figures 6.1, and 6.2, we have to use a standalone visualisation script called sampling_visualisation_script.py to combine the sampling test results.

To use this script, we can call it in the command line:
`python sampling_visualisation_script.py -s "METHOD" -p /path/to/that/samplingmethod/test.csv {repeated for each sampling method}`

<sub>It might look something like this:
`python sampling_visualisation_script.py --paths ./files/sampling_tests/full_data/test_896811.csv -s uniform -p ./files/sampling_tests/full_data/test_3697990.csv -s cluster -o ./files/sampling_tests -p ./files/sampling_tests/full_data/test_8375147.csv -s empirical`<sub>

## Chapter 7

#### Sections 7.1.1 - 7.1.3
Normalisation is covered by the `normalisation_tool.py` script, which enables you to normalise the trill .csvs files from `./files/data_processed`. It also contains all the object oriented code for normalising, which is then used in the other scripts for this section.

#### Figure 7.1
The anchor transition varience visualisation is done by `anchor_trills_visualiser.py`. We can call it as follows:

`python anchor_trills_visualiser.py -d ./path/to/submission_data/Processed_Data.csv -o ./path/to/output.png`. 

For this script we have to use the `Processed_Data.csv` file.

#### Figure 7.2, and 7.3
The normalisation visualiser is done by `normalisation_visualiser.py`. We can call it as follows:

`python ./normalisation_visualiser.py -d ./path/to/submission_data/Processed_Data.csv -i ./files/normalised/AdditiveNorm.csv -o ./output.png -n additive -s 0.2`

for Figure 7.2.

And:

`python ./normalisation_visualiser.py -d ./submission_data/Processed_Data.csv -i ./files/normalised/MultiplicativeNorm.csv -o ./output.png -n multiplicative -s 0.2`

for Figure 7.3.

#### Table 7.1 (Section 7.1.4)
The normalisation tests are handled by the `normalisation_evaluation.py` script.

We can run them as follows:

`python normalisation_evaluation.py --csvs ./files/data_processed/ -o ./path/to/output.txt --strength 0.2 --norm_style {additive|multiplicative}`

Default seeding was used.

#### Section 7.2.1 - 7.2.4
The code for personalisation is inside `personalisation_test.py`.

#### Table 7.2

To generate the data from Table 7.2, we can run the following:

`python personalisation_test.py --csvs ./files/data_processed/ -o ./path/to/output.txt --strength 0.2 --norm_style {additive|multiplicative}`

## Chapter 8
The difficulty model code is contained within `difficulty_estimator.py` (using fingering prediction code from `fingering_prediction.py`). To use this difficulty estimation using a model trained from the data we have, we can type the following on to the command line:

The code uses the E-HB (No EW) feature set.

Short_Segment_Annotated.mxl was generated with BPM set to 160.

Both other examples generated with BPM set to 200.

#### Figure 8.1
To generate the MusicXML file for Figure 8.1, we can do the following:

`python difficulty_estimator.py -i ./files/examples/Short_Segment.mxl -o ./files/examples/Short_Segment_Annotated.mxl -d /path/to/data/Processed_Data.csv -e ./encodings/encodings.txt --bpm 160`

The actual image is an export of the MusicXML file using Musescore.
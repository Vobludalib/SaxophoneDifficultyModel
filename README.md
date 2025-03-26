## **How to recreate figures:**

#### Installing requirements
TODO: Create requirements.txt
pip install requirements.txt

TODO: Make all file paths relative or CL based

#### encoding.py
encoding.py is used as a library throughout the other Python scripts, as it unifies operations of transitions, encodings, and feature extraction methods using an object-based approach.

#### Generating the sessions
Todo:
CL

#### Prompt generation (figure 1)

Todo:
1) Debug session generator (make difficulty scaling different?)
2) Prompt generator trill numbers fix
3) Writeup music21 requirements (I use Musescore 3 for the rendering, possible additional setup may be needed). Link music21 documentation
4) CL

#### Trill detection procedure
1) Use trill_detection_tool.py to get trills .csv file
2) Use batch_analysis_tool.py to verify trill detection

Todo:
Add trill speed to batch_analysis_tool.py readout for easier verification
CL

Explain how crepe salience plot is obtained via command line

#### Model tests
TODO:
1) Show exact procedure for generating the table + graph
2) CL for paths

#### Sampling tests
TODO:
1) CL 
2) Show procedure for generating the figures (assurance of run time)
   1) sampling.py -> save files -> use visualiser scripts

#### Difficulty model
TODO:
1) CL
2) Provide the file to reproduce result
3) Generate result again (with fixed seeding)
# Plugin specific documentation

I have used my Musescore Python script plugin generator (see [here](https://github.com/Vobludalib/MusescoreIntegration) for fuller documentation) to create this plugin.

This plugin allows you to use the model visualiser directly inside Musescore 3.6 on Windows.

To install this plugin, copy the ```DifficultyModel``` directory into your Musescore Plugins directory (see [here](https://musescore.org/en/handbook/3/plugins)). For technical reasons, this plugin will work with Musescore 3.6 and Windows, other setups are not supported.

For a demo of the plugin, see ```PluginDemo.gif```, which demonstrates not just the saxophone difficulty visualiser, but other toy difficulty models, showing how this script can easily be expanded.

To run it, your global Python installation that is called using the ```python``` command on the command line must have all the requirements from ```requirements.txt``` installed. 

For anyone wishing to isolate these dependencies into a seperate environment, note that the command-line call actually takes place in the Musescore ```bin``` folder. I have not tested isolating the environment, so you will have some exploration and debugging to do.

The plugin works by calling ```model_to_visualisation.py``` with appropriate command line arguments from inside a Musescore plugin, sending it a MusicXML file and then receiving the colored in MusicXML file. 

## What models are included
Right now, the only actual model is the saxophone model from the rest of this repository. Two toy examples which assign either random or pitch-dependent difficulties to notes are present to show that this plugin is expandable.

## Adding models
You will need to add the Python script containing the code for creating/loading and running the model into the ```DifficultyModel``` directory, and then import it into the ```model_to_visualisation.py``` script. You will have to change the switch logic and argument expansion in the ```main``` method of said script to incorporate your script. Inspiration can be taken from the existing saxophone model implementation.

You will then have to add this model as an option in the .qml file of the plugin itself. Notably, you will have to add it to the ```onRun``` ```flags``` variable if you want to modify default behaviour and add it as an option to the ComboBox. This process should be fairly self-explanatory. If needed, consult the documentation for the plugin generator.

## Debugging your Python script
Debugging the Python script when it is called from the Musescore plugin can be a pain. I recommend running the plugin from within the Musescore plugin creator, as this has a console to which the plugin logs certain things.

Primarily, the log shows the actual command line call that the script is being invoked with. Remember that the Python script is actually being invoked from Musescore's ```bin``` directory, so relative file paths may require additional scrutiny.

If an exception is thrown, it is printed to the standard output of the script, and thus forwarded to the plugin to write to console. Hopefully, these debugging tools are satisfactory for any competent programmer.
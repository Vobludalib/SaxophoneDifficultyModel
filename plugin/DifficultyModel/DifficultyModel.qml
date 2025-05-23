import QtQuick 2.1
import QtQuick.Dialogs 1.2
import QtQuick.Controls 1.0
import MuseScore 3.0
import FileIO 3.0
import Qt.labs.folderlistmodel 2.2
import Qt.labs.platform 1.0

MuseScore {
	menuPath: "Plugins.DifficultyModel"
	version: "0.1"
	id: window
	width: 800; height: 500;
	requiresScore: true
	pluginType: "dialog"

	QProcess {
		id: proc
	}

property var flags: {}

onRun: {
    flags = {  "--model": { toPrint: true, value: "saxophone" }, "--bpm": { toPrint: true, value: "120" }, "--easy_color": { toPrint: true, value: "#000000" }, "--hard_color": { toPrint: true, value: "#FF0000" }}
    loadingText.visible = false;
    executableScript.source = getLocalPath(executableScript.source);
    mscTempFileStorePath.source = getLocalPath(mscTempFileStorePath.source);
}

TextField {
    id: option0
    placeholderText: qsTr("120")
    anchors.top: window.top
    anchors.left: window.left
    anchors.topMargin: 10
    anchors.leftMargin: 10
    anchors.bottomMargin: 10
    width: 150
    height: 30
    onEditingFinished: {
        if (option0.text != "") {
            flags["--bpm"].value = option0.text;
        }
    }
}

Text {
    id: option0Prompt
    text: qsTr("What is the BPM? Enter a number")
    anchors.left: option0.right
    anchors.top: option0.top
    anchors.bottom: option0.bottom
    anchors.leftMargin: 10
}
TextField {
    id: option1
    placeholderText: qsTr("#000000")
    anchors.top: option0.bottom
    anchors.left: window.left
    anchors.topMargin: 10
    anchors.leftMargin: 10
    anchors.bottomMargin: 10
    width: 150
    height: 30
    onEditingFinished: {
        flags["--easy_color"].value = option1.text;
    }
}

Text {
    id: option1Prompt
    text: qsTr("What is the color for 'easy' notes? Enter a hex value including the # (e.g. #FF0000)")
    anchors.left: option1.right
    anchors.top: option1.top
    anchors.bottom: option1.bottom
    anchors.leftMargin: 10
}
TextField {
    id: option2
    placeholderText: qsTr("#FF0000")
    anchors.top: option1.bottom
    anchors.left: window.left
    anchors.topMargin: 10
    anchors.leftMargin: 10
    anchors.bottomMargin: 10
    width: 150
    height: 30
    onEditingFinished: {
        if (option2.text != "") {
            flags["--hard_color"].value = option2.text;
        }
    }
}

Text {
    id: option2Prompt
    text: qsTr("What is the color for 'hard' notes? Enter a hex value including the # (e.g. #FF0000)")
    anchors.left: option2.right
    anchors.top: option2.top
    anchors.bottom: option2.bottom
    anchors.leftMargin: 10
}

ComboBox {
    id: option3
    currentIndex: 2
    width: 200
    anchors.top: option2.bottom
    anchors.left: window.left
    anchors.topMargin: 10
    anchors.leftMargin: 10
    anchors.bottomMargin: 10
    model: ListModel {
        id: option3Model
        ListElement { text: "Random"; value: "random" }
        ListElement { text: "Pitch"; value: "pitch" }
        ListElement { text: "Saxophone"; value: "saxophone" }
    }
    onCurrentIndexChanged: {
        var val = option3Model.get(currentIndex).value;
        changeoption3Value(val);
    }
}

Text {
    id: option3Prompt
    text: qsTr("What model should be used?")
    anchors.left: option3.right
    anchors.top: option3.top
    anchors.bottom: option3.bottom
    anchors.leftMargin: 10
}

function changeoption3Value(val) {
    flags["--model"].value = val;
}

FileIO {
    id: executableScript
    source: "./model_to_visualisation.py"
    onError: console.log(msg)
}

FileIO {
    id: mscTempFileStorePath
    source: "./temp/temp"
}

Text {
	id: loadingText
	text: qsTr("LOADING!!!")
	anchors.centerIn: window
}

Button {
	id: buttonSave
	text: qsTr("Launch executable")
	anchors.bottom: window.bottom
	anchors.left: window.left
	anchors.topMargin: 10
	anchors.bottomMargin: 10
	anchors.leftMargin: 10

	onClicked: {
		var call = createCLICallFromFlags();
		console.log(call);
		proc.start(call);
		loadingText.visible = true;
		var val = proc.waitForFinished(30000);
		loadingText.visible = false;

		var output = proc.readAllStandardOutput();
		console.log("Finished python script with output: " + output);
		
		var correctOutputPath = getLocalPath(String(output));
		console.log('"' + correctOutputPath + '"');
		readScore(correctOutputPath);
		
	}
}

function createCLICallFromFlags() {
    var call = "python";
    call = call + ' "' + executableScript.source + '"';
    
    var tempFilePath = mscTempFileStorePath.source + getCurrentTimeString();
    writeScore(curScore, tempFilePath, "mxl");
    print("Saved score to " + tempFilePath + ".mxl");
    call = call + ' --tempPath "' + tempFilePath + '.mxl"';
    
    for (var key in flags) {
        if (flags[key].toPrint) {
            call = call + " " + key;
            if (flags[key].value != "") {
                call = call + ' "' + flags[key].value + '"';
            }
        }
    }

    call = call + " 2>&1";

    return call;
}

function getLocalPath(path) { // Remove "file://" from paths and third "/" from  paths in Windows
    path = path.trim();
    path = path.replace(/^(file:\/{2})/,"");

    var pluginsFolder = window.filePath;
    path = path.replace(/^\./, pluginsFolder);

    if (Qt.platform.os == "windows") { 
        path = path.replace(/^\//,"");
        path = path.replace(/\//g, "\\");
    }
    path = decodeURIComponent(path);            
    return path;
}

function getCurrentTimeString() {
    var cT = new Date();
    var timeString = cT.toLocaleDateString(Qt.locale("en_CA"), Locale.ShortFormat);
    timeString = timeString + "-" + cT.getHours() + "-" + cT.getMinutes() + "-" + cT.getSeconds();
    return timeString;
}

}
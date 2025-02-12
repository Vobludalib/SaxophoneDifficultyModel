"""
LH-----------_RH--------
TP--MIY---A--_PMI--Y-A--

Oct, Front F, LFF, Bis, LSF, LTF, G#, C#, B, Bb, High F, High D#, High D _ RFF, RSF, High F#, Side F#, RTF, Eb, C, High E, Side C, Side Bb
"""

from enum import Enum
import csv
import itertools
import numpy as np

print_debug = False

def main():
    fingerings = load_fingerings_from_file("/Users/slibricky/Desktop/Thesis/thesis/encodings.txt")

def load_fingerings_from_file(file_path, delim=','):
    fingerings = []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=delim)
        next(reader, None)
        for row in reader:
            midi = int(row[0].strip())
            name = row[1].strip()
            encoding = row[2].strip()
            if print_debug: print(name)
            fingerings.append(Fingering(midi, name, encoding))

    return fingerings

# Loads a list of transitions with their trill speed from a csv file
# CSV FILE IS EXPECTED TO HAVE HEADER:
# Filename,Cluster,Midi 1 (transposed as written for TS),Fingering 1 Name,Fingering 1 Encoding,Midi 2 (transposed as written for TS),Fingering 2 Name,Fingering 2 Encoding,Trill Speed
def load_transitions_from_file(file_path, delim=','):
    transitions_trills_dict = {}
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=delim)
        next(reader, None)
        for row in reader:
            midi1 = int(row[2].strip())
            name1 = row[3].strip()
            encoding1 = row[4].strip()
            midi2 = int(row[5].strip())
            name2 = row[6].strip()
            encoding2 = row[7].strip()
            trill_speed = float(row[8].strip())
            note1 = Fingering(midi1, name1, encoding1)
            note2 = Fingering(midi2, name2, encoding2)
            trans = Transition(note1, note2)
            if transitions_trills_dict.get(trans, None) is None:
                transitions_trills_dict[trans] = [trill_speed]
            else:
                transitions_trills_dict[trans].append(trill_speed)

    return transitions_trills_dict

def generate_interval_features(interval):
    fingering1, fingering2 = interval
    # Delta in finger changes on each hand (i.e. finger pressing something then not pressing or vice verse)
    finger_changes_per_hand = []
    for handi in range(2):
        finger_changes = 0
        for fingeri in range(5):
            finger1 = fingering1.hands[handi].fingers[fingeri]
            finger2 = fingering2.hands[handi].fingers[fingeri]
            if True in [False if finger1.keys[i].pressed == finger2.keys[i].pressed else True for i in range(len(finger1.keys))]:
                finger_changes += 1
        
        finger_changes_per_hand.append(finger_changes)

    if print_debug: print(f"\t# of fingers changing per hand: {finger_changes_per_hand}")

    # Delta in octave key
    octave_key_delta = True if fingering1.hands[0].fingers[0].keys[0].pressed != fingering2.hands[0].fingers[0].keys[0].pressed else False
    if print_debug: print(f"\tOctave key change: {"True" if octave_key_delta else "False"}")

    # Amount of same-finger transitions (except same-hand palm)
    same_finger_transitions = 0
    for handi in range(2):
        finger_changes = 0
        for fingeri in range(5):
            finger1 = fingering1.hands[handi].fingers[fingeri]
            finger2 = fingering2.hands[handi].fingers[fingeri]

            finger1pressed = [True if key.pressed else False for key in finger1.keys]
            finger2pressed = [True if key.pressed else False for key in finger2.keys]
            mismatches = 0
            is_lifting = None
            for i in range(len(finger1pressed)):
                if finger1pressed[i] != finger2pressed[i]:
                    if is_lifting is None:
                        if finger1pressed[i] and not finger2pressed[i]:
                            is_lifting = True
                        else:
                            is_lifting = False
                    else:
                        if is_lifting == (finger1pressed[i] and not finger2pressed[i]):
                            continue
                    
                    # do not increment on Bis key
                    for key1index in range(len(finger1.keys)):
                        if (finger1.keys[key1index].name == Keys.Bis):
                            if (finger1.keys[key1index].pressed != finger2.keys[key1index].pressed):
                                continue
                    mismatches += 1
            
            if (mismatches > 1):
                same_finger_transitions += 1
                if print_debug: print(f"\t--- SFT ON {fingering1.hands[handi].side}-{finger1.name}")

    if print_debug: print(f"\t# of same-finger transitions: {same_finger_transitions}")

    fingering1_l_palm_pressed = any([key.pressed for key in fingering1.hands[0].fingers[5].keys])
    fingering1_r_palm_pressed = any([key.pressed for key in fingering1.hands[1].fingers[5].keys])
    fingering2_l_palm_pressed = any([key.pressed for key in fingering2.hands[0].fingers[5].keys])
    fingering2_r_palm_pressed = any([key.pressed for key in fingering2.hands[1].fingers[5].keys])
    change_palm_l = 10 if fingering1_l_palm_pressed != fingering2_l_palm_pressed else 0
    change_palm_r = 10 if fingering1_r_palm_pressed != fingering2_r_palm_pressed else 0
    contains_low_cs_or_lower = 5*(fingering1.midi <= 61 or fingering2.midi <= 61)

    return f"{fingering1.name} to {fingering2.name}", np.asarray([fingering1.midi / 10, fingering2.midi / 10, contains_low_cs_or_lower, abs(fingering1.midi - fingering2.midi), finger_changes_per_hand[0], finger_changes_per_hand[1], 10 if octave_key_delta else 0, same_finger_transitions*20, change_palm_l, change_palm_r])

class Hands(Enum):
    LEFT = 0
    RIGHT = 1

class Fingers(Enum):
    THUMB = 0
    POINTER = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4
    PALM = 5

class Keys(Enum):
    Oct = 0
    FrontF = 1
    LeftFirstFront = 2
    Bis = 3
    LeftSecondFront = 4
    LeftThirdFront = 5
    Gs = 6
    LowCs = 7
    LowB = 8
    LowBb = 9
    HighF = 10
    HighDs = 11
    HighD = 12
    RightFirstFront = 13
    RightSecondFront = 14
    HighFs = 15
    SideFs = 16
    RightThirdFront = 17
    Eb = 18
    LowC = 19
    HighE = 20
    SideC = 21
    SideBb = 22

class Fingering:
    midi = 0
    name = ""
    hands = []

    def __init__(self, midi: int, name: str, encoding: str):
        self.midi = midi
        self.name = name
        self.hands = [self.Hand(Hands.LEFT), self.Hand(Hands.RIGHT)]
        self.read_encoding(encoding)

    def read_encoding(self, encoding: str):
        sides = encoding.split(sep="_")
        encoding = sides[0] + sides[1]
        i = 0
        for hand in self.hands:
            for finger in hand.fingers:
                for key in finger.keys:
                    key.pressed = True if encoding[i] == "1" else False
                    i += 1

    def generate_encoding(self):
        output = ""
        i = 0
        last_hand = None
        for hand in self.hands:
            if last_hand is not None and last_hand.side == Hands.LEFT and hand.side == Hands.RIGHT:
                output += "_"
            last_hand = hand
            for finger in hand.fingers:
                for key in finger.keys:
                    output += "1" if key.pressed else "0"
                    i += 1

        return output

    def __str__(self):
        output = f"{self.name}"
        # for hand in self.hands:
        #     for finger in hand.fingers:
        #         for key in finger.keys:
        #             output += f"{key.name}: {key.pressed}\n"
        
        return output

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Fingering):
            return self.generate_encoding() == other.generate_encoding()
        else:
            return NotImplemented
        
    def __hash__(self):
        return self.generate_encoding().__hash__()

    class Key:
        name = None
        parent_finger = None
        pressed = False

        def __init__(self, name, parent_finger, pressed = False):
            self.name = name
            self.parent_finger = parent_finger
            self.pressed = pressed

    class Finger:
        name = None
        keys = []
        parent_hand = None

        def __init__(self, name, parent_hand):
            self.name = name
            self.parent_hand = parent_hand

            if (parent_hand.side == Hands.LEFT):
                match self.name:
                    case Fingers.THUMB:
                        self.keys = [Fingering.Key(Keys.Oct, self)]
                    case Fingers.POINTER:
                        self.keys = [Fingering.Key(Keys.FrontF, self), Fingering.Key(Keys.LeftFirstFront,self), Fingering.Key(Keys.Bis, self)]
                    case Fingers.MIDDLE:
                        self.keys = [Fingering.Key(Keys.LeftSecondFront, self)]
                    case Fingers.RING:
                        self.keys = [Fingering.Key(Keys.LeftThirdFront, self)]
                    case Fingers.PINKY:
                        self.keys = [Fingering.Key(Keys.Gs, self), Fingering.Key(Keys.LowCs, self), Fingering.Key(Keys.LowB, self), Fingering.Key(Keys.LowBb, self)]
                    case Fingers.PALM:
                        self.keys = [Fingering.Key(Keys.HighF, self), Fingering.Key(Keys.HighDs, self), Fingering.Key(Keys.HighD, self)]
            else:
                match self.name:
                    case Fingers.THUMB:
                        self.keys = []
                    case Fingers.POINTER:
                        self.keys = [Fingering.Key(Keys.RightFirstFront, self)]
                    case Fingers.MIDDLE:
                        self.keys = [Fingering.Key(Keys.RightSecondFront, self)]
                    case Fingers.RING:
                        self.keys = [Fingering.Key(Keys.HighFs, self), Fingering.Key(Keys.SideFs, self), Fingering.Key(Keys.RightThirdFront, self)]
                    case Fingers.PINKY:
                        self.keys = [Fingering.Key(Keys.Eb, self), Fingering.Key(Keys.LowC, self)]
                    case Fingers.PALM:
                        self.keys = [Fingering.Key(Keys.HighE, self), Fingering.Key(Keys.SideC, self), Fingering.Key(Keys.SideBb, self)]

    class Hand:
        side = None
        fingers = {}

        def __init__(self, side):
            self.side = side
            self.fingers = [
                Fingering.Finger(Fingers.THUMB, self), 
                Fingering.Finger(Fingers.POINTER, self), 
                Fingering.Finger(Fingers.MIDDLE, self), 
                Fingering.Finger(Fingers.RING, self), 
                Fingering.Finger(Fingers.PINKY, self), 
                Fingering.Finger(Fingers.PALM, self)]

class Transition:
    def __init__(self, fing1, fing2):
        self.fingering1 = fing1
        self.fingering2 = fing2

    def __eq__(self, other):
        return self.fingering1 == other.fingering1 and self.fingering2 == other.fingering2
    
    def __hash__(self):
        return self.fingering1.__hash__() + self.fingering2.__hash__()
    
    def __repr__(self):
        return f"{self.__hash__()} -> {self.fingering1.name} to {self.fingering2.name}"
         
if __name__ == "__main__":
    main()
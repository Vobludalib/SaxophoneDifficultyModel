"""
LH-----------_RH--------
TP--MIY---A--_PMI--Y-A--

Oct, Front F, LFF, Bis, LSF, LTF, G#, C#, B, Bb, High F, High D#, High D _ RFF, RSF, High F#, Side F#, RTF, Eb, C, High E, Side C, Side Bb
"""

from enum import Enum
import csv
import itertools
import numpy as np
from abc import ABC, abstractmethod

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

def generate_all_transitions(fingerings):
    transitions = []
    for comb in itertools.combinations(fingerings, 2):
        transitions.append(Transition(comb[0], comb[1]))
    
    return transitions

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
    palm_keys_tied_to_fingers = False
    key_to_hand_finger = {}

    def __init__(self, midi: int, name: str, encoding: str, tie_palm_keys_to_fingers = False):
        self.midi = midi
        self.name = name
        self.hands = [self.Hand(Hands.LEFT), self.Hand(Hands.RIGHT)]
        self.read_encoding(encoding)
        self.palm_keys_tied_to_fingers = tie_palm_keys_to_fingers
        self.key_to_hand_finger = {
            Keys.HighF : (Hands.LEFT, Fingers.MIDDLE),
            Keys.HighDs: (Hands.LEFT, Fingers.POINTER),
            Keys.HighD: (Hands.LEFT, Fingers.POINTER),
            Keys.HighE: (Hands.RIGHT, Fingers.POINTER),
            Keys.SideC: (Hands.RIGHT, Fingers.POINTER),
            Keys.SideBb: (Hands.RIGHT, Fingers.POINTER)
        }

        if tie_palm_keys_to_fingers:
            self.convert_palm_as_finger_to_no_palm()

    def read_encoding(self, encoding: str):
        sides = encoding.split(sep="_")
        encoding = sides[0] + sides[1]
        i = 0
        for hand in self.hands:
            for finger in hand.fingers:
                for key in finger.keys:
                    key.pressed = True if encoding[i] == "1" else False
                    i += 1

    def convert_palm_as_finger_to_no_palm(self):
            # NOTE: This keeps the palm 'Finger' still present, so make sure to not iterate over all fingers
            for handi in range(len(self.hands)):
                palm_keys = self.hands[handi].fingers[5].keys
                for key in palm_keys:
                    correct_handi, correct_fingeri = self.key_to_hand_finger[key.name]
                    self.hands[correct_handi.value].fingers[correct_fingeri.value].keys.append(key)

            self.palm_keys_tied_to_fingers = True

    def convert_no_palm_to_palm_as_finger(self):
            for handi in range(2):
                palm_keys = self.hands[handi].fingers[5].keys
                for key in palm_keys:
                    correct_handi, correct_fingeri = self.key_to_hand_finger[key.name]
                    self.hands[correct_handi.value].fingers[correct_fingeri.value].keys.pop()

            self.palm_keys_tied_to_fingers = False

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
                    # Checking if we have palm keys in places where they aren't expected
                    if self.palm_keys_tied_to_fingers:
                        if finger.name != Fingers.PALM and (key in self.hands[0].fingers[5].keys or key in self.hands[1].fingers[5].keys):
                            continue
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
        return f"{self.fingering1.name} to {self.fingering2.name}"
    
    def __getitem__(self, index):
        if index == 0:
            return self.fingering1
        elif index == 1:
            return self.fingering2
        else:
            NotImplementedError

class TransitionFeatureExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_features(self, transition: Transition) -> np.ndarray:
        pass

class RawFeatureExtractor(TransitionFeatureExtractor):
    def __init__(self):
        pass

    def get_features(self, transition: Transition) -> np.ndarray:
        fingering1 = transition[0]
        fingering2 = transition[1]
        encoding1 = fingering1.generate_encoding()
        encoding2 = fingering2.generate_encoding()
        features = []
        for encoding in (encoding1, encoding2):
            for val in encoding:
                if val == '_':
                    continue
                else:
                    features.append(int(val))
        
        return np.asarray(features)

class FingerFeatureExtractor(TransitionFeatureExtractor):
    same_finger_transition_weight = 10

    def __init__(self):
        pass

    def get_features(self, transition: Transition) -> np.ndarray:
        """
        Used to get features based on if a given finger has to move during the transition + added penalty for same finger transition
        """
        fingering1 = transition[0]
        fingering2 = transition[1]

        for fingeringi in range(2):
            if not transition[fingeringi].palm_keys_tied_to_fingers:
                transition[fingeringi].convert_palm_as_finger_to_no_palm()

        features = []
        for handi in range(2):
            for fingeri in range(5):
                finger1 = fingering1.hands[handi].fingers[fingeri]
                finger2 = fingering2.hands[handi].fingers[fingeri]
                does_change = False
                if True in [False if finger1.keys[i].pressed == finger2.keys[i].pressed else True for i in range(len(finger1.keys))]:
                    does_change = True
                features.append(1 if does_change else 0)

        same_finger_transitions = 0
        for handi in range(2):
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
        
        features.append(same_finger_transitions * self.same_finger_transition_weight)

        return np.asarray(features)

class ExpertFeatureExtractor(TransitionFeatureExtractor):
    palm_weight = 10
    low_key_weight = 5
    midi_weight = 0.1
    octave_key_weight = 10
    same_finger_transition_weight = 20 
    finger_transition_weight = 1
    leap_size_weight = 1

    def __init__(self, use_expert_weights=True, remove_midi=False):
        self.use_expert_weights=use_expert_weights
        self.remove_midi = remove_midi

    def get_features(self, transition: Transition) -> np.ndarray:
        fingering1 = transition[0]
        fingering2 = transition[1]

        for fingeringi in range(2):
            if transition[fingeringi].palm_keys_tied_to_fingers:
                transition[fingeringi].convert_no_palm_to_palm_as_finger()

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

        # Delta in octave key
        octave_key_delta = True if fingering1.hands[0].fingers[0].keys[0].pressed != fingering2.hands[0].fingers[0].keys[0].pressed else False

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

        # Values for if a palm key state is swapped for each hand
        fingering1_l_palm_pressed = any([key.pressed for key in fingering1.hands[0].fingers[5].keys])
        fingering1_r_palm_pressed = any([key.pressed for key in fingering1.hands[1].fingers[5].keys])
        fingering2_l_palm_pressed = any([key.pressed for key in fingering2.hands[0].fingers[5].keys])
        fingering2_r_palm_pressed = any([key.pressed for key in fingering2.hands[1].fingers[5].keys])
        change_palm_l = (self.palm_weight if self.use_expert_weights else 1) if fingering1_l_palm_pressed != fingering2_l_palm_pressed else 0
        change_palm_r = (self.palm_weight if self.use_expert_weights else 1) if fingering1_r_palm_pressed != fingering2_r_palm_pressed else 0
        # Checking if a low note appears (as this influences voicing significantly)
        contains_low_cs_or_lower = (fingering1.midi <= 61 or fingering2.midi <= 61) * (self.low_key_weight if self.use_expert_weights else 1)
        # Does octave key have to change state
        octave_key_value = (self.octave_key_weight if self.use_expert_weights else 1) if octave_key_delta else 0
        same_finger_transition_value = same_finger_transitions*(self.same_finger_transition_weight if self.use_expert_weights else 1)
        leap_size_value = self.leap_size_weight * abs(fingering1.midi - fingering2.midi)

        no_midi_features = [leap_size_value, contains_low_cs_or_lower, self.finger_transition_weight*finger_changes_per_hand[0], self.finger_transition_weight*finger_changes_per_hand[1], octave_key_value, same_finger_transition_value, change_palm_l, change_palm_r]

        if self.remove_midi:
            return np.asarray(no_midi_features)
        else:
            midi1 = fingering1.midi
            midi2 = fingering2.midi
            return np.asarray([midi1/self.midi_weight, midi2/self.midi_weight] + no_midi_features)

if __name__ == "__main__":
    main()

import numpy as np
import music21

def stringify(val):
    return ','.join([str(x) for x in val])


def flatten(arr):
    return [i for j in arr for i in j]


def update_corpus(corp, key):
    if key in corp:
        corp[key] += 1
    else:
        corp[key] = 1
    return corp

def update_key_vec(key, melodies, corp):
    if key not in corp:
        corp[key] = dict()

    for m in melodies:
        for n in m:
            pitch = n.pitch.midi
            while pitch - 12 >= 0:
                pitch -= 12
            if pitch in corp[key]:
                corp[key][pitch] += n.quarterLength
            else:
                corp[key][pitch] = n.quarterLength
    return corp


def update_pitch_vec(chords, melodies, corp):
    for index, c in enumerate(chords):
        c = stringify(c)
        if c not in corp:
            corp[c] = dict()
        for n in melodies[index]:
            pitch = n.pitch.midi
            if pitch in corp[c]:
                corp[c][pitch] += n.quarterLength
            else:
                corp[c][pitch] = n.quarterLength
    return corp


def melody_pitch_encoding(melodies):
    exact = []
    for melody in melodies:
        pitches = []
        for i in range(len(melody)):
            if type(melody[i]) is music21.note.Rest:
                continue
            pitches.append(melody[i].pitch.midi)
        exact.append(pitches)
    return exact


def build_corp_reduced(c, m):
    m = melody_pitch_encoding(m)
    # convert melody to interval representation
    relative = []
    for melody in m:
        temp_interval = []
        for i in range(len(melody) - 1):
            temp_interval.append(melody[i + 1] - melody[i])
        relative.append(temp_interval)
    return build_duo_corp([c], [relative])


def remove_rests(melody):
    return [note for note in melody if type(note) is not music21.note.Rest]


def update_duo_corp(chords, melodies, corp):
    for chord in chords:
        chord = stringify(chord)
        if chord not in corp:
            corp[chord] = dict()

        for melody in melodies:
            melody = stringify(melody)
            if melody in corp[chord]:
                corp[chord][melody] += 1
            else:
                corp[chord][melody] = 1
    return corp



def build_duo_corp(c, m):

    m = flatten(m)
    c = flatten(c)
    # for each of chord; build corpus of melodies, then combine to make multi-layerd dict.
    # lookup dictionary; chord -> index of that chord in the temporary list of melody dictionaries
    c_lookup = dict()
    corp = []
    for index, chord in enumerate(c):

        chord = stringify(chord)

        # if chord currently has a corpus built from it; find from lookup table
        if chord in c_lookup:
            lookup_index = c_lookup[chord]
            m_key = stringify(m[index])

            corp[lookup_index] = update_corpus(corp[lookup_index], m_key)

        # if chord does not currently have a corpus; create corp
        else:
            # create corp
            n = len(c_lookup)
            c_lookup[chord] = n
            corp.append(dict())
            m_key = stringify(m[index])
            corp[n] = update_corpus(corp[n], m_key)

    c_corpus = dict()
    for index, key in enumerate(c_lookup):
        c_corpus[key] = corp[index]

    return c_corpus

def build_single_corp(c):
    c = flatten(c)
    corp = dict()
    for index, chord in enumerate(c):
        chord = stringify(chord)
        corp = update_corpus(corp, chord)
    return corp

def convert_corpus_to_probabilities(corp):
    for key in corp:
        total = sum(corp[key].values())
        for sub_key in corp[key]:
            corp[key][sub_key] = float(corp[key][sub_key] / total)
    return corp

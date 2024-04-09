import music21
from music21 import stream
import numpy as np
from sklearn import preprocessing
from utils import flatten

CONS_VEC = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0]


def chord_melody_separation(score):
    melodies = []
    score = [n for n in score.recurse().notes]
    separation = [x for x in range(len(score)) if type(score[x]) == music21.harmony.ChordSymbol]
    chords = [score[i] for i in separation]
    separation.append(len(score))
    for i in range(len(separation) - 1):
        melodies.append(score[separation[i] + 1:separation[i + 1]])

    return chords, melodies


def chord_extract(chords, key):
    # normal orders of chords
    normal_orders = []
    # chords rotated to pitch class 0
    pc0 = []
    numerals = []
    for c in chords:
        norm = c.normalOrder
        fp = norm[0]
        rotate = [(n - fp) % 12 for n in norm]
        normal_orders.append(norm)
        pc0.append(rotate)

        #  generate GCT
        numeral = music21.roman.romanNumeralFromChord(c, key)
        # print(numeral.figure)
        numerals.append(numeral.figure)

    return normal_orders, pc0, numerals


def melody_extract(melodies):
    pitched = []
    intervals = []
    for m in melodies:
        p = [n.pitch.midi for n in m]
        inter = []
        for i in range(len(p) - 1):
            inter.append(p[i + 1] - p[i])

        pitched.append(p)
        intervals.append(inter)

    return pitched, intervals

def pitchweight_extract(melodies):
    weighted = []
    for m in melodies:
        note_dict = dict()
        total = 0
        wm = []

        for index, n in enumerate(m):
            pitch = n.pitch.midi
            if pitch in note_dict:
                note_dict[pitch] += n.duration.quarterLength
            else:
                note_dict[pitch] = n.duration.quarterLength
            total += 1

        for key in note_dict:
            note_dict[key] /= total

        for n in m:
            wm.append(note_dict[n.pitch.midi])
        weighted.append(wm)
    return weighted


def melodic_reduction(melodies, pitched, intervals, pitch_weights, ratio=0.75):
    reduced = False
    original = 0

    salience_arr = []
    pitch_stab_arr = []
    beat_strength_arr = []
    duration_arr = []

    # calculate individual note-weights
    for i, m in enumerate(melodies):
        w = []
        reduction_weights = []
        saliences = []
        pitch_stabs = []
        beat_strengths = []
        durations = []
        for j, n in enumerate(m):
            if j == 0:
                interval_salience = 0
            else:
                interval_salience = np.abs(intervals[i][j - 1])

            pitch_stability = pitch_weights[i][j]
            beat_strength = n.beatStrength
            duration = n.duration.quarterLength

            saliences.append(interval_salience)
            pitch_stabs.append(pitch_stability)
            beat_strengths.append(beat_strength)
            durations.append(duration)

            original += 1

        # saliences = np.array(saliences)
        # pitch_stabs = np.array(pitch_stabs)
        # beat_strengths = np.array(beat_strengths)
        # durations = np.array(durations)

        salience_arr.append(saliences)
        pitch_stab_arr.append(pitch_stabs)
        beat_strength_arr.append(beat_strengths)
        duration_arr.append(durations)


    flat_salience_arr = flatten(salience_arr)
    flat_pitch_stab_arr = flatten(pitch_stab_arr)
    flat_beat_strength_arr = flatten(beat_strength_arr)
    flat_duration_arr = flatten(duration_arr)
    # normalize all arrays to [0, 1]


    min_sal = np.min(flat_salience_arr)
    max_sal = np.max(flat_salience_arr)

    min_stab = np.min(flat_pitch_stab_arr)
    max_stab = np.max(flat_pitch_stab_arr)

    min_bs = np.min(flat_beat_strength_arr)
    max_bs = np.max(flat_beat_strength_arr)

    min_d = np.min(flat_duration_arr)
    max_d = np.max(flat_duration_arr)

    for i in range(len(salience_arr)):
        salience_arr[i] = np.array(salience_arr[i])
        salience_arr[i] = (salience_arr[i] - min_sal) / (max_sal - min_sal)

        pitch_stab_arr[i] = np.array(pitch_stab_arr[i])
        pitch_stab_arr[i] = (pitch_stab_arr[i] - min_stab) / (max_stab - min_stab)

        beat_strength_arr[i] = np.array(beat_strength_arr[i])
        beat_strength_arr[i] = (beat_strength_arr[i] - min_bs) / (max_bs - min_bs)

        duration_arr[i] = np.array(duration_arr[i])
        duration_arr[i] = (duration_arr[i] - min_d) / (max_d - min_d)

    reduction_weights = [0] * len(salience_arr)
    for i in range(len(salience_arr)):
        salience_arr[i] = salience_arr[i].tolist()
        pitch_stab_arr[i] = pitch_stab_arr[i].tolist()
        beat_strength_arr[i] = beat_strength_arr[i].tolist()
        duration_arr[i] = duration_arr[i].tolist()

        reduction_weights[i] = np.add(reduction_weights[i], salience_arr[i])
        reduction_weights[i] = np.add(reduction_weights[i], pitch_stab_arr[i])
        reduction_weights[i] = np.add(reduction_weights[i], beat_strength_arr[i])
        reduction_weights[i] = np.add(reduction_weights[i], duration_arr[i])


    for t in range(400):
        thresh = t * 0.01
        temp = []
        new_count = 0
        for i, m in enumerate(reduction_weights):
            rm = []
            for j, weight in enumerate(m):
                if weight > thresh:
                    rm.append(pitched[i][j])
                    new_count += 1
            temp.append(rm)

        r = (new_count / original)
        if r < ratio:
            reduced = True
            break

    if reduced:
        reduction = temp

    else:


        reduction = pitched

    intervals = []
    for m in reduction:
        intervals.append([m[i + 1] - m[i] for i in range(len(m) - 1)])

    return reduction, intervals


def melodic_reduction_test(melodies, pitched, intervals, pitch_weights, ratio=0.25):
    reduced = False
    original = 0

    salience_arr = []
    pitch_stab_arr = []
    beat_strength_arr = []
    duration_arr = []

    # calculate individual note-weights
    for i, m in enumerate(melodies):
        w = []
        reduction_weights = []
        saliences = []
        pitch_stabs = []
        beat_strengths = []
        durations = []
        for j, n in enumerate(m):
            if j == 0:
                interval_salience = 0
            else:
                interval_salience = np.abs(intervals[i][j - 1])

            pitch_stability = pitch_weights[i][j]
            beat_strength = n.beatStrength
            duration = n.duration.quarterLength

            saliences.append(interval_salience)
            pitch_stabs.append(pitch_stability)
            beat_strengths.append(beat_strength)
            durations.append(duration)

            original += 1

        # saliences = np.array(saliences)
        # pitch_stabs = np.array(pitch_stabs)
        # beat_strengths = np.array(beat_strengths)
        # durations = np.array(durations)

        salience_arr.append(saliences)
        pitch_stab_arr.append(pitch_stabs)
        beat_strength_arr.append(beat_strengths)
        duration_arr.append(durations)


    flat_salience_arr = flatten(salience_arr)
    flat_pitch_stab_arr = flatten(pitch_stab_arr)
    flat_beat_strength_arr = flatten(beat_strength_arr)
    flat_duration_arr = flatten(duration_arr)
    # normalize all arrays to [0, 1]


    min_sal = np.min(flat_salience_arr)
    max_sal = np.max(flat_salience_arr)

    min_stab = np.min(flat_pitch_stab_arr)
    max_stab = np.max(flat_pitch_stab_arr)

    min_bs = np.min(flat_beat_strength_arr)
    max_bs = np.max(flat_beat_strength_arr)

    min_d = np.min(flat_duration_arr)
    max_d = np.max(flat_duration_arr)

    for i in range(len(salience_arr)):
        salience_arr[i] = np.array(salience_arr[i])
        salience_arr[i] = (salience_arr[i] - min_sal) / (max_sal - min_sal)

        pitch_stab_arr[i] = np.array(pitch_stab_arr[i])
        pitch_stab_arr[i] = (pitch_stab_arr[i] - min_stab) / (max_stab - min_stab)

        beat_strength_arr[i] = np.array(beat_strength_arr[i])
        beat_strength_arr[i] = (beat_strength_arr[i] - min_bs) / (max_bs - min_bs)

        duration_arr[i] = np.array(duration_arr[i])
        duration_arr[i] = (duration_arr[i] - min_d) / (max_d - min_d)

    reduction_weights = [0] * len(salience_arr)
    for i in range(len(salience_arr)):
        salience_arr[i] = salience_arr[i].tolist()
        pitch_stab_arr[i] = pitch_stab_arr[i].tolist()
        beat_strength_arr[i] = beat_strength_arr[i].tolist()
        duration_arr[i] = duration_arr[i].tolist()

        reduction_weights[i] = np.add(reduction_weights[i], salience_arr[i])
        reduction_weights[i] = np.add(reduction_weights[i], pitch_stab_arr[i])
        reduction_weights[i] = np.add(reduction_weights[i], beat_strength_arr[i])
        reduction_weights[i] = np.add(reduction_weights[i], duration_arr[i])


    for t in range(400):
        thresh = t * 0.01
        temp = []
        new_count = 0
        for i, m in enumerate(reduction_weights):
            rm = []
            for j, weight in enumerate(m):
                if weight > thresh:
                    rm.append(melodies[i][j])
                    new_count += 1
                else:
                    rest = music21.note.Rest()
                    rest.duration = melodies[i][j].duration
                    rm.append(rest)

            temp.append(rm)

        r = (new_count / original)
        print(new_count)
        if r < ratio:
            reduced = True
            break


    if reduced:
        reduction = temp

    else:
        print("No reduction found")
        reduction = pitched

    return reduction


def extract(score):
    key = score.analyze('key')
    chords, melodies = chord_melody_separation(score)
    normal_order, pc0, numerals = chord_extract(chords, key)
    pitched, intervals = melody_extract(melodies)
    pitch_weights = pitchweight_extract(melodies)
    reduction, interval_reduction = melodic_reduction(melodies, pitched, intervals, pitch_weights)

    return chords, melodies, normal_order, pc0, numerals, pitched, intervals, pitch_weights, reduction, interval_reduction


def reconstruct(score, reduced):
    # score = [n for n in score.recurse().notesAndRests]
    # flat_melody = [n for m in reduced for n in m]
    # for index, note in enumerate(score):
    #     if type(note) is music21.note.Note and type(note) is not music21.note.Rest:
    #         if flat_melody:
    #             n = flat_melody.pop(0)
    #             n = music21.note.Note(n, quarterLength=note.duration.quarterLength)
    #         else:
    #             n = music21.note.Rest(quarterLength=note.duration.quarterLength)
    #         score[index] = n
    flat = [n for m in reduced for n in m]
    S = stream.Stream()
    for i in flat:
        S.append(i)

    return S

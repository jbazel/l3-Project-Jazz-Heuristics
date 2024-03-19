import music21
import numpy as np

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


def melodic_reduction(melodies, pitched, intervals, pitch_weights, ratio=0.7):
    reduced = False
    original = 0
    reduction_weights = []
    # calculate individual note-weights
    for i, m in enumerate(melodies):
        w = []
        for j, n in enumerate(m):
            if j == 0:
                interval_salience = 0
            else:
                interval_salience = np.abs(intervals[i][j - 1])

            pitch_stability = pitch_weights[i][j]
            beat_strength = n.beatStrength
            duration = n.duration.quarterLength
            c = interval_salience + pitch_stability + beat_strength + duration
            w.append(c)
            original += 1

        reduction_weights.append(w)

    for t in range(200):
        thresh = t * 0.025
        temp = []
        new_count = 0
        for i, m in enumerate(reduction_weights):
            rm = []
            for j, weight in enumerate(m):
                if weight > thresh:
                    rm.append(pitched[i][j])
                    new_count += 1
            temp.append(rm)

        r = np.abs((new_count / original) - ratio)
        if r < 0.05:
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


def extract(score):
    key = score.analyze('key')
    chords, melodies = chord_melody_separation(score)
    normal_order, pc0, numerals = chord_extract(chords, key)
    pitched, intervals = melody_extract(melodies)
    pitch_weights = pitchweight_extract(melodies)
    reduction, interval_reduction = melodic_reduction(melodies, pitched, intervals, pitch_weights)

    return chords, melodies, normal_order, pc0, numerals, pitched, intervals, pitch_weights, reduction, interval_reduction

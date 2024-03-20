import music21
import numpy as np
import random
import json
from feature_extraction import extract
from search import three_gram_search


# error_amnt is a float value representing what proporition of the notes to "error"
# goal is to shift notes off by a random amount -> this will not always provide an error that is "off key"
# but will give a fair representation of what an omr algorithms error could produce
#



def add_errors(score, err_amnt, file_name):
    chords, \
        melodies, \
        normal_order, \
        pc0, \
        numerals, \
        pitched, \
        intervals, \
        pitch_weights, \
        reduction, \
        interval_reduction = extract(score)

    # first add errors to pitched melodies

    key = score.analyze('key')
    notes = [n for n in score.recurse().notes]
    len_notes = len(notes)

    flags = []
    # get the indices of the notes to error
    for melody_ind, m in enumerate(pitched):
        temp = []
        for note_ind, n in enumerate(m):
            if random.random() < err_amnt:
                n += random.choice([-1, 1])  # add a random amount to the note
                pitched[melody_ind][note_ind] = n
                temp.append(False)
            else:
                temp.append(True)
        flags.append(temp)

    # next recompute the intervals
    intervals = []
    for m in pitched:
        intervals.append([m[i + 1] - m[i] for i in range(len(m) - 1)])

    # now we need to recompute the reduction

    three_gram_search(pitched, intervals, pitch_weights, True, pc0, intervals)

add_errors(music21.converter.parse('EWLD/dataset/Adam_Anders-Nikki_Hassman-Peer_Åström/If_Only/If_Only.mxl'), 0.4, 'test_data/1.xml')

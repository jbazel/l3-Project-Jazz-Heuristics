import music21
import numpy as np
import random
import json
from feature_extraction import extract
from search import three_gram_search
import sqlite3

# error_amnt is a float value representing what proporition of the notes to "error"
# goal is to shift notes off by a random amount -> this will not always provide an error that is "off key"
# but will give a fair representation of what an omr algorithms error could produce
#

db = sqlite3.connect('EWLD/EWLD.db')
cursor = db.cursor()
cursor.execute(
    'SELECT DISTINCT t.path_leadsheet FROM works t INNER JOIN work_genres w ON t.id = w.id WHERE w.genre = "Jazz"')
paths = cursor.fetchall()
n = int(len(paths) -len(paths) / 10)

paths = paths[n:]

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
    # randomly shift notes up and down one semitone
    for melody_ind, m in enumerate(pitched):
        temp = []
        for note_ind, n in enumerate(m):
            if random.random() < err_amnt:
                n += random.choice([-1, 1])  # add a random amount to the note
                pitched[melody_ind][note_ind] = n
                temp.append(1)
            else:
                temp.append(0)
        flags.append(temp)

    # random note addition
    for melody_ind, m in enumerate(pitched):

        length = len(m)
        points = [1 if random.random() < 0.1 else 0 for i in range(length)]
        for i in range(len(points)):
            if points[i] == 1:
                note = random.randint(min(m) - 12, max(m) + 12)
                pitched[melody_ind].insert(i, note)
                flags[melody_ind].insert(i, 1)


      # random note deletion
    for melody_ind, m in enumerate(pitched):
        for i in range(4):
            if m:
                if random.random() < 0.1:
                    to_delete = random.randint(0, len(m)-1)
                    pitched[melody_ind].pop(to_delete)
                    flags[melody_ind].pop(to_delete)

    # next recompute the intervals
    intervals = []
    for m in pitched:
        intervals.append([m[i + 1] - m[i] for i in range(len(m) - 1)])

    with open(file_name, "w+") as f:
        json.dump({"pitched": pitched, "intervals": intervals, "normal_order": normal_order, "pc0": pc0, "flags": flags}, f)

counter = 0
for path in paths:
    counter+=1
    path = "EWLD/" + path[0]
    score = music21.converter.parse(path)
    # if len(score.recurse().getElementsByClass(meter.TimeSignature)) == 0:
    #     print("ERROR: Invalid Time Signature - Skipping")
    #     continue
    add_errors(score, 0.4, "test_files/" + str(counter) + ".json")

import numpy as np
import music21
from feature_extraction import extract
import json
from nltk import edit_distance
import statistics


def stringify(val):
    return ','.join([str(x) for x in val])


def three_gram_search(score, corpus, note_probabilities, flag):
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

    if flag == 0:
        chord_set = normal_order
        melody_set = pitched

    elif flag == 1:
        chord_set = numerals
        melody_set = intervals

    else:
        print("ERROR: Invalid flag")
        return

    probs = []
    for index, m in enumerate(melody_set):
        length = len(m)
        if length > 3:
            for i in range(length - 2):
                search_gram = m[i:i + 3]
                current_chord = chord_set[index]
                current_chord = stringify(current_chord)
                if current_chord in corpus:
                    mindx = 100000
                    for comparison_gram in corpus[current_chord].keys():
                        dx = edit_distance(search_gram, comparison_gram)
                        if dx < mindx:
                            mindx = dx
                            if dx != 0:
                                prob = corpus[current_chord][comparison_gram] * (1 - (0.01 / dx))
                            else :
                                prob = corpus[current_chord][comparison_gram]
                            probs.append([search_gram, prob])
                    print("current search gram {} has probability: {}".format(search_gram, prob))

                else:
                    average_note_prob = 0
                    chord_key = stringify(normal_order[index])
                    for note in pitched[index]:
                        note = str(note)
                        if note in note_probabilities[chord_key]:
                            average_note_prob += note_probabilities[chord_key][note]
                    average_note_prob /= len(search_gram)
                    prob = average_note_prob
                    print("current search gram {} has probability: {}".format(search_gram, prob))
                    probs.append([search_gram, prob])

        else:
            if not m:
                continue
            search_gram = m
            current_chord = chord_set[index]
            current_chord = stringify(current_chord)

            if current_chord in corpus:
                mindx = 100000
                for comparison_gram in corpus[current_chord].keys():
                    dx = edit_distance(search_gram, comparison_gram)
                    # if dx != 0:
                    #     prob = corpus[current_chord][comparison_gram] * (1 / dx)
                    # else:
                    #     prob = corpus[current_chord][comparison_gram]

                    if dx < mindx:
                        mindx = dx
                        prob = corpus[current_chord][comparison_gram]
                        probs.append([search_gram, prob])

                print("current search gram {} has probability: {}".format(search_gram, prob))

            else:
                average_note_prob = 0
                chord_key = stringify(normal_order[index])

                for note in pitched[index]:
                    note = str(note)
                    if note in note_probabilities[chord_key]:
                        average_note_prob += note_probabilities[chord_key][note]
                average_note_prob /= len(search_gram)
                prob = average_note_prob
                print("current search gram {} has probability: {}".format(search_gram, prob))

                probs.append([search_gram, prob])

    return probs


def compare(gram, corpus, chord):
    mindx = 100000
    for comparison_gram in corpus[chord].keys():
        dx = edit_distance(gram, comparison_gram)
        if dx < mindx:
            mindx = dx
            if dx != 0:
                prob = corpus[chord][comparison_gram] * (1 - (0.01 / dx))
            else :
                prob = corpus[chord][comparison_gram]
    return prob, len(gram)





def three_gram_search_test(pitched, intervals, normal_order, numerals, corpus, note_probabilities, flag, key):
    start_note = 0
    if flag == 0:
        chord_set = normal_order
        melody_set = pitched

    elif flag == 1:
        # potentially change this back
        # chord_set = numerals
        chord_set = numerals
        melody_set = intervals

    else:
        print("ERROR: Invalid flag")
        return

    probs = []
    for index, m in enumerate(melody_set):
        low = 0
        ciel = len(m)
        high = 3 if len(m) > 2 else len(m) - 1

        while low < ciel + 1:
            search_gram = m[low:high]
            if not search_gram:
                low += 1
                continue
            current_chord = chord_set[index]
            current_chord = stringify(current_chord)
            if current_chord in corpus:
                prob, n = compare(search_gram, corpus, current_chord)
                probs.append([prob, [start_note +i for i in range(n)]])
                start_note+=1
            else:
                average_note_prob = 0
                for note in search_gram:
                    while note - 12 >= 0:
                        note -= 12
                    note = str(note)
                    if note in note_probabilities[key]:
                        average_note_prob += note_probabilities[key][note]
                    else:
                        average_note_prob += 0
                average_note_prob /= len(search_gram)
                prob = average_note_prob

                probs.append([prob, [start_note+i for i in range(3)]])
                start_note+=1

            low += 1
            high = high + 1 if (high + 1 < ciel) else ciel

    return probs

def analyse_prob_dist(dist, num_notes, thresh = 3):
    probs = [x[0] for x in dist]
    mean = statistics.mean(probs)
    if len(probs) > 1:
        variance = statistics.variance(probs)
    else:
        variance = 0.0001
    z_scores = [(x - mean) / variance for x in probs]
    flags =np.array([0]*num_notes)
    # print(z_scores)
    for i in range(len(z_scores)):
        if z_scores[i] < -thresh:
            if np.array(dist[i][1]).any() >= len(flags) - 2:
                flags[-1] += 1

            else:
                try:
                    flags[dist[i][1]] += 1
                except:
                    print(dist[i][1], len(flags))
    # print([dist[i][1] for i in range(len(dist))])
    # print(flags)
    return flags

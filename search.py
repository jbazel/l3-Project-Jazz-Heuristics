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
                        # if dx != 0:
                        #     prob = corpus[current_chord][comparison_gram] * (1 - (0.1 / dx))
                        # else:
                        #     prob = corpus[current_chord][comparison_gram]
                        if dx < mindx:
                            mindx = dx
                            prob = corpus[current_chord][comparison_gram]

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

    return probs, len(pitched)


def three_gram_search_test(pitched, intervals, normal_order, pc0, corpus, note_probabilities, flag):
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
                        # if dx != 0:
                        #     prob = corpus[current_chord][comparison_gram] * (1 - (0.1 / dx))
                        # else:
                        #     prob = corpus[current_chord][comparison_gram]
                        if dx < mindx:
                            mindx = dx
                            prob = corpus[current_chord][comparison_gram]

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

    return probs, len(pitched)

def analyse_prob_dist(dist, num_notes):
    probs = [x[1] for x in dist]
    mean = statistics.mean(probs)
    variance = statistics.variance(probs)
    z_scores = [(x - mean) / variance for x in probs]


    flags = [0]*num_notes
    for i in range(len(z_scores)):
        if z_scores[i] > 30:
            for j in range(len(dist[0])):
                flags[i+j] += 1


    # for i in range(len(dist)):
    #     print("Z-score for {} is: {}".format(dist[i][0], z_scores[i]))

    print(flags)
    return flags


score = music21.converter.parse('EWLD/dataset/Adam_Anders-Nikki_Hassman-Peer_Åström/If_Only/If_Only.mxl')
probs_pitched, n = three_gram_search(score, json.load(open('corpi/reduced_pitched_corpus.json')), json.load(open('corpi/pitch_vec.json')), 0)
probs_unpitched, n = three_gram_search(score, json.load(open('corpi/reduced_relative_corpus.json')), json.load(open('corpi/pitch_vec.json')), 1)

analyse_prob_dist(probs_pitched, n)
analyse_prob_dist(probs_unpitched, n)

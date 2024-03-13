import numpy as np
import music21
from feature_extraction import extract
import json
from nltk import edit_distance


def three_gram_search(score, corpus, note_probabilities):
    reduced_pitched_corp = json.load(open("corpi/reduced_pitched_corpus.json"))
    reduced_relative_corp = json.load(open("corpi/reduced_relative_corpus.json"))
    corpus = reduced_relative_corp
    chords, \
    melodies, \
    normal_order, \
    pc0, \
    pitched, \
    intervals, \
    pitch_weights, \
    reduction, \
    interval_reduction = extract(score)

    for index, m in enumerate(pitched):
        length = len(m)
        if length > 3:
            for i in range(length - 2):
                search_gram = m[i:i+3]
                current_chord = pc0[index]
                if current_chord in corpus:
                    current = corpus[current_chord]
                    for comparison_melody in current.keys():
                        if search_gram in comparison_melody:
                            print("found: ", search_gram, " in ", comparison_melody)
                        else:
                            min_dist = 100000
                            prob = 0
                            # find nearest match
                            for comparison in current.keys:
                                dist = edit_distance(search_gram, comparison)
                                if dist < min_dist:
                                    min_dist = dist
                                    prob = current[comparison]
                                    print("nearest match: ", comparison, " with distance: ", dist)


                average_note_prob = 0
                for note in search_gram:
                    average_note_prob += note_probabilities[current_chord][note]
                average_note_prob /= len(search_gram)



def search(score, method_flag=0):
    reduced_pitched_corp = json.load(open("corpi/reduced_pitched_corpus.json"))
    reduced_relative_corp = json.load(open("corpi/reduced_relative_corpus.json"))
    corpus = reduced_relative_corp
    chords, \
    melodies, \
    normal_order, \
    pc0, \
    pitched, \
    intervals, \
    pitch_weights, \
    reduction, \
    interval_reduction = extract(score)

    bar_flags = [False for i in range(len(reduction))]

    for i, m in enumerate(interval_reduction):
        prob = 0
        min_edit = 100000
        chord = pc0[i]
        if chord in corpus:
            current = corpus[chord]
            for comparison_melody in current.keys:
                # distance is levenstein
                edit = edit_distance(reduction[i], comparison_melody)
                if edit < min_edit:
                    min_edit = edit
                    prob = current[comparison_melody]

        print(prob)


def search_witout_chord(search_melody, corpus):

    pass

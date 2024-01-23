class ChordType:
    def __init__(self, chord):
        self.type = chord
        self.corp = dict()

    def add_to_dict(self, melody):
        melody = [str(i) for i in melody]
        melody = ",".join(melody)
        if melody in self.corp:
            self.corp[melody] += 1
        else:
            self.corp[melody] = 1


def build_chord_interval_corpus(chords, melodies):
    chord_dict = dict()
    for i in range(len(chords)):
        if len(chords[i]) != len(melodies[i]):
            print("err: missmatch length")
            return

        for j in range(len(chords[i])):
            if chords[i][j] not in chord_dict:
                chord_dict[chords[i][j]] = ChordType(chords[i][j])
            chord_dict[chords[i][j]].add_to_dict(melodies[i][j])

    return chord_dict
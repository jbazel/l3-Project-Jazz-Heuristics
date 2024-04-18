import music21
import sqlite3
import random
import os
def get_samples():
    db = sqlite3.connect('EWLD/EWLD.db')
    cursor = db.cursor()
    cursor.execute(
        'SELECT DISTINCT t.path_leadsheet FROM works t INNER JOIN work_genres w ON t.id = w.id WHERE w.genre = "Jazz"')
    paths = cursor.fetchall()
    db.close()

    samples = random.sample(paths, 10)
    return samples

def open_for_photo(samples):
    paths = []
    for i in range(len(samples)):
        path = "EWLD/" + samples[i][0]
        score = music21.converter.parse(path)
        path = score.write('musicxml.png')
        os.rename(str(path), 'omr_evaluation_images/image{}.png'.format(i))
samples = get_samples()
open_for_photo(samples)

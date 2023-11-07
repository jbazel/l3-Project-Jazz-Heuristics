import music21.converter
from music21 import *
import sqlite3
us = environment.UserSettings()
us['musicxmlPath'] ='../../../../../Applications/MuseScore 3.app'
us['midiPath'] ='../../../../../Applications/GarageBand.app'
c = music21.converter.parse("EWLD/dataset/Alicia_Scott/Annie_Laurie/Annie_Laurie.mxl")

db = sqlite3.connect('EWLD/EWLD.db')
cursor = db.cursor()
cursor.execute('SELECT * FROM "work_genres" where "genre" = "Jazz"')
print(len(cursor.fetchall()))

# # lyric removal
# for n in c.recurse().notesAndRests:
#     if n.lyric:
#         print(n.lyric)
#         n.lyric = ''
#
# # chord removal
# for n in c.recurse().getElementsByClass(chord.Chord):
#     meas = n.getContextByClass(stream.Measure)
#     print(meas.measureNumber, meas.offset, n.getOffsetInHierarchy(c), n.offset)
#     meas.remove(n)




# c.show()


# db = "/OpenEWLD-master/OpenEWLD.db"
# db = sqlite3.connect(db)


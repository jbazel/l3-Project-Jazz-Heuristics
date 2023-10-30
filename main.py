import music21.converter
from music21 import *
import sqlite3
us = environment.UserSettings()
us['musicxmlPath'] ='../../../../../Applications/MuseScore 4.app'
us['midiPath'] ='../../../../../Applications/GarageBand.app'
c = music21.converter.parse("OpenEWLD-master/dataset/Fats_Waller/The_Jitterbug_Waltz/The_Jitterbug_Waltz.mxl")
c.show('MIDI'
)

#
# db = "/OpenEWLD-master/OpenEWLD.db"
# db = sqlite3.connect(db)


print([x.id for x in features.extractorsById('all')])
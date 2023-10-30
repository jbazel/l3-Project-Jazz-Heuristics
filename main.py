import music21.converter
from music21 import *
c = music21.converter.parse("OpenEWLD-master/dataset/Alicia_Scott/Annie_Laurie/Annie_Laurie.mxl")
c.show("Annie_Laurie.pdf")
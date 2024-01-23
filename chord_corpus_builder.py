"""
PROPOSED STAGES:
1. Start with a maximal "search size"
2. Iterate through a progression decreasing the "search size"
3. Store pattern matches to that progression - patterns are described as within an edit-distance threshold
4. Take the n-most maximal patterns found
5. Repeat for all progressions returning the n-most maximal patterns
"""

import nltk
import numpy as np

# potentially alter minimum edit distance based on disparity between length?

def buildCorpus(chords):
    corp = dict()
    for i in chords:
        for j in i:
            j = stringify(j)
            if j not in corp:
                corp[j] = 1
            else:
                corp[j] +=1
    return corp

def stringify(val):
    return ','.join([str(x) for x in val])



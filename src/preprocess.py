import numpy as np
import pandas as pd
import io
import re
from collections import Counter

import pickle


PUNCT_TAG = "__PUNCT__"


def clean_and_split(corpus):
    aux = re.sub('<.*?>', '', corpus)
    aux = re.sub('<.*?>', '', aux)
    aux = re.sub('\[.*?\]', '', aux)
    aux = re.sub('  +', ' ', aux)
    aux = re.sub('\n\n+','\n\n', aux)
    return aux.split("\n\n")



#Read full file
file = io.open("CETENFolha-1.0_jan2014.cg", mode="r", encoding="utf-8")
file.seek(0)
corpus = file.read()

# Clean input text
sentences = clean_and_split(corpus)


# Select (word,token) tuples
phrases = [[
        (l.split("\t")[0].lower().strip() if len(l.split("\t")) > 1 else l.split(" ")[0],
        l.split("\t")[1].split(" ")[1] if len(l.split("\t")) > 1 else PUNCT_TAG)
        for l in s.split("\n")] 
        for s in list(filter(None, sentences))]


# Allowed tags
allowed_tags = ['N','DET','PRP','V','PROP','ADJ','ADV','NUM','KC','SPEC','PERS','KS','IN','EC','PRON',PUNCT_TAG]

# Remove tags not allowed (with small frequency)
phrases = (list(filter(lambda s: all(tk[1] in allowed_tags for tk in s), phrases)))


with open('preprocessed_CETEN_v2.pkl', 'wb') as output:
    pickle.dump(phrases, output, pickle.HIGHEST_PROTOCOL)


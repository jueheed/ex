'''
Created on Mar. 24, 2020

@author: hong
'''
import spacy
from pprint import pprint
from collections import defaultdict

# Load English tokenizer, tagger, parser, NER and word vectors and more
nlp = spacy.load("en_core_web_lg")

# Read whole documents
f=open('data/data3.txt', "r")
text =f.read()

with nlp.disable_pipes('ner'):
    doc = nlp(text)

threshold = 0.2
(beams) = nlp.entity.beam_parse([ doc ], beam_width = 16, beam_density = 0.0001)

entity_scores = defaultdict(float)
for beam in beams:
    for score, ents in nlp.entity.moves.get_beam_parses(beam):
        for start, end, label in ents:
            entity_scores[(start, end, label)] += score

print ('Entities and scores (detected with beam search)')
for key in entity_scores:
    start, end, label = key
    score = entity_scores[key]
    if ( score > threshold):
        #print out entity type, text, probability
        print ('({}: {}: {})'.format(doc[start:end], label, score))

# Find named entities, phrases and concepts
#pprint([(entity.text, entity.label_) for entity in doc.ents])

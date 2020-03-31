'''
Created on Mar. 24, 2020

@author: hong
'''
import spacy
from pprint import pprint
from collections import defaultdict

class DefaultNerEx:
   def ner_search(self,file):
      # Load English
      nlp = spacy.load('en_core_web_lg')

      # Read whole documents
      f=open(file, "r")
      message =f.read()

      with nlp.disable_pipes('ner'):
           doc = nlp(message)

      threshold = 0.2
      (beams) = nlp.entity.beam_parse([ doc ], beam_width = 16, beam_density = 0.0001)

      entity_scores = defaultdict(float)
      for beam in beams:
          for score, ents in nlp.entity.moves.get_beam_parses(beam):
              for start, end, label in ents:
                  entity_scores[(start, end, label)] += score

      print ('Entities and scores (detected with beam search)')
      entities=[]
      for key in entity_scores:
          start, end, label = key
          score = entity_scores[key]
          if ( score > threshold):
               entities.append((doc[start:end], label, score))
      return entities

if __name__ == '__main__':
   #execute the action
   entities=DefaultNerEx().ner_search('data/data1.txt')

   # Find named entities with text, entity type, probability
   pprint([(entity) for entity in entities])

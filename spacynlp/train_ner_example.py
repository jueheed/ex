'''
Created on Mar. 26, 2020

@author: hong
'''
import spacy
import random
from pprint import pprint


class TrainNerEx:
   #TRAIN_DATA= [('I am writing to report a deal which has aroused my suspicion who does a money laundering.', {'entities': [(72, 88, 'KW')]}), ('The fraudulent from the release of the deposit would be handled by the lawyers.', {'entities': [(4, 14, 'KW')]}), ('The fraud was also investigated by police but there is no result.', {'entities': [(4, 9, 'KW'), (35, 41, 'KW')]})]
   TRAIN_DATA= [('Introduction:Internal Case #C92646759\nFinTRAC Reference Number: 348729667', {'entities': [(38, 45, 'ORG')]}), ('On 1/12/2017 Capital One Bank USA, NA (COBUSANA) detected patterns of activity on credit card account number 1234567890123456', {'entities': [(3, 12, 'DATE'), (13, 33, 'ORG'), (35, 37, 'GPE')]}), ('Capital One has identified an unknown suspect', {'entities': [(0, 11, 'ORG')]}), ('All supporting documentation is maintained by Capital One Bank (USA) N.A.', {'entities': [(46, 62, 'ORG')]}), ('The suspicious activity was conducted in Canada with Walmart, restaurants, Target, Quality Inn, Best Buy and other brick and mortar merchants', {'entities': [(41, 47, 'GPE'), (53, 60, 'ORG'), (75, 81, 'ORG'), (83, 94, 'ORG'), (96, 104, 'ORG')]}), ('From 3/15/2016 to 5/14/2016 suspicious activity occurred totaling $8,098.76', {'entities': [(5, 14, 'DATE'), (18, 27, 'DATE'), (66, 75, 'MONEY')]}), ('The sample is drawn from transaction activity occurring from 5/12/2016 to 5/14/2016', {'entities': [(61, 70, 'DATE'), (74, 83, 'DATE')]}), ('Included in Section B of this report is a sampling of transactional activity totaling $30.0', {'entities': [(86, 91, 'MONEY')]})]

   def train_spacy(self, data, iters):

          TRAIN_DATA = data
          # create blank Language class
          nlp = spacy.blank('en')
    
          # create the built-in pipeline components and add them to the pipeline
          # nlp.create_pipe works for built-ins that are registered with spaCy
          if 'ner' not in nlp.pipe_names:
              ner = nlp.create_pipe('ner')
              nlp.add_pipe(ner, last=True)
       
          # add labels
          for _, annotations in TRAIN_DATA:
               for ent in annotations.get('entities'):
                  ner.add_label(ent[2])

          # get names of other pipes to disable them during training
          other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
          with nlp.disable_pipes(*other_pipes):  # only train NER
              optimizer = nlp.begin_training()
              for iter in range(iters):
                  print("Starting iteration " + str(iter))
                  random.shuffle(TRAIN_DATA)
                  losses = {}
                  for text, annotations in TRAIN_DATA:
                      nlp.update(
                          [text],  # batch of texts
                          [annotations],  # batch of annotations
                          drop=0.2,  # dropout - make it harder to memorise data
                          sgd=optimizer,  # callable to update weights
                          losses=losses)
                  print(losses)
          return nlp
      
   def ner_search(self,file):
      # Read whole documents
      f=open(file, "r")
      message=f.read()
      TRAIN_DATA = self.TRAIN_DATA
      nlp =  self.train_spacy(TRAIN_DATA,20)


      model_file = 'custommodel'
      # Save our trained Model
      nlp.to_disk(model_file)

      #Test each sentence from train data
      for text, _ in TRAIN_DATA:
          nlp(text)

      nlp = spacy.load(model_file)
      doc = nlp(message)

      entities=[]
      for ent in doc.ents:
          entities.append((ent.text, ent.label_))
      return entities
      

      
if __name__ == '__main__':
   entities=TrainNerEx().ner_search('data/data1.txt')
   
   # Find named entities from the message
   pprint([(entity) for entity in entities])
   
   #for entity in doc.ents:
   #    print(entity.text, entity.label_)

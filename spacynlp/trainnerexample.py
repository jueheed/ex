import spacy
import random
from pprint import pprint


# Read whole documents
f=open('data/traindata.txt', "r")
message =f.read()

TRAIN_DATA = [('I am writing to report a deal which has aroused my suspicion who does a money laundering.', {'entities': [(72, 88, 'KW')]}), ('The fraudulent from the release of the deposit would be handled by the lawyers.', {'entities': [(4, 14, 'KW')]}), ('The fraud was also investigated by police but there is no result.', {'entities': [(4, 9, 'KW'), (35, 41, 'KW')]})]


def train_spacy(data,iters):
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
        for itn in range(iters):
            print("Statring iteration " + str(itn))
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

prdnlp = train_spacy(TRAIN_DATA, 20)

# Save our trained Model
modelfile = 'custommodel'
#
prdnlp.to_disk(modelfile)

#Test each sentense from train data
for text, _ in TRAIN_DATA:
    doc = prdnlp(text)

nlp = spacy.load(modelfile)
doc = prdnlp(message)

# Find named entities from the message
pprint([(entity.text, entity.label_) for entity in doc.ents])
#for entity in doc.ents:
#    print(entity.text, entity.label_)

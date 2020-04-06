import flask
from flask import request, jsonify
from default_ner_example import DefaultNerEx

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

dataFile = 'data/data1.txt'

# A route to return all of the available entries from NER search.
@app.route('/api/v1/resources/entities/all', methods=['GET'])
def get_all_entities():
    lst=DefaultNerEx().ner_search(dataFile)
    keys=['entity_name', 'label']
    entity_list = []
    for ent in lst:
           ent1=[]
           ent1.append(str(ent[0]))
           ent1.append(ent[1])
           entity_list.append(dict(zip(keys,ent1)))

    return jsonify(entity_list)

# A route to return all of the available entries from NER search.
@app.route('/api/v1/resources/entities', methods=['GET'])
def get_entities_by_label():
    # Check if an label was provided as part of the URL.
    # If label is provided, assign it to a variable.
    # If no label is provided, display an error in the browser.
    if 'label' in request.args:
        label = str(request.args['label'])
    else:
        return "Error: No label field provided. Please specify an label."
    lst=DefaultNerEx().ner_search(dataFile)
    keys=['entity_name', 'label']
    entity_list = []
    for ent in lst:
        if ent[1] == label.upper():
           ent1=[]
           ent1.append(str(ent[0]))
           ent1.append(ent[1])
           entity_list.append(dict(zip(keys,ent1)))
    return jsonify(entity_list)

app.run()

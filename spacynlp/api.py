import flask
import json
from default_ner_example import DefaultNerEx

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    lst=DefaultNerEx().ner_search('data/data1.txt')
    listToStr = ' '.join([str(elem) for elem in lst])
    #list_json_string = json.dumps(lst)
    return listToStr

app.run()

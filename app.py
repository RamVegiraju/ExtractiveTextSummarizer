from flask import Flask, request, jsonify, render_template
import pandas as pd
from textsummary import freq_calculator, find_entities, show_entities
import requests
import spacy 
from spacy import displacy
nlp = spacy.load('en_core_web_sm')


app = Flask(__name__)

@app.route('/', methods = ["GET","POST"]) 
def index():
    if request.method == "POST":
        input_text = request.form.get('url')
        #text_content = freq_calculator(input_text)
        summary = freq_calculator(input_text)
        #entities = show_entities(summary)
        return summary
    return render_template("index.html")

#@app.route('/NER', methods=["GET","POST"])
#def namedEntity():
    #if request.method == "POST":
        #input_text = request.form.get('entrecognition')
        #entities = show_entities(input_text)
        #return entities
    #return render_template("tables.html", tables = [entities.to_html(classes='data')], titles = entities.columns.values)


if __name__ =='__main__':
    app.run(debug=True)
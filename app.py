from flask import Flask, request, jsonify, render_template
import pandas as pd
from textsummary import freq_calculator, find_entities, show_entities
import requests
import spacy 
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
from flaskext.markdown import Markdown


app = Flask(__name__)
Markdown(app)

#HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

@app.route('/', methods = ["GET","POST"]) 
def index():
    if request.method == "POST":
        input_text = request.form.get('url')
        #text_content = freq_calculator(input_text)
        summary = freq_calculator(input_text)
        return summary
    return render_template("index.html")

@app.route('/NER', methods=["GET","POST"])
def namedEntity():
    if request.method == "POST":
        rawtext = request.form['rawtext']
        docs = nlp(rawtext)
        result = displacy.render(docs,style='ent')
        #result = result.replace("\n\n","\n")
        #result = HTML_WRAPPER.format(html)
    return render_template("results.html", rawtext = rawtext, result=result)


if __name__ =='__main__':
    app.run(debug=True)
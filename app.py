from flask import Flask, request, jsonify, render_template
from textsummary import freq_calculator, find_entities, show_entities
import requests


app = Flask(__name__)

@app.route('/', methods = ["GET","POST"]) 
def index():
    if request.method == "POST":
        input_text = request.form.get('url')
        #text_content = freq_calculator(input_text)
        summary = freq_calculator(input_text)
        entities = show_entities(summary)
        return summary, entities
    return render_template("index.html")
    #return '<h1>FlASK APP IS RUNNING</h1>'

if __name__ =='__main__':
    app.run(debug=True)























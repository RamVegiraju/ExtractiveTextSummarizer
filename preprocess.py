import pandas as pd 
import numpy as np
import re
import six
import nltk
import spacy
import string
import ast
import inflect
import unicodedata
import matplotlib.pyplot as plt
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
file = open("contractions.txt", "r")
contents = file.read()
#print("All good right now")
contraction_map = eval(contents)
file.close()


class textPreProcessor():

    def __init__(self,text):
        self.text = text

    #Expanding Contractions
    def expand_contractions(self, contraction_mapping=contraction_map):
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())                       
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction
        
        expanded_text = contractions_pattern.sub(expand_match, self.text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text
    
    #Converting to lowercase
    def text_lowercase(self): 
        return self.text.lower()
    
    #Converting numbers to words
    def convert_number(self): 
        p = inflect.engine() 
        temp_str = self.text.split() 
        new_string = [] 
    
        for word in temp_str: 
            if word.isdigit(): 
                temp = p.number_to_words(word) 
                new_string.append(temp) 
            else: 
                new_string.append(word) 
        temp_str = ' '.join(new_string) 
        return temp_str 
    
    # remove punctuation 
    def remove_punctuation(self): 
        translator = str.maketrans('', '', string.punctuation) 
        return self.text.translate(translator)
    
    # remove whitespace from text 
    def remove_whitespace(self): 
        return  " ".join(self.text.split())

    #Lemmatization
    def word_lemmatizer(self):
        lemmatizer = WordNetLemmatizer()
        lem_text = [lemmatizer.lemmatize(i) for i in self.text]
        lemmas = ''.join(lem_text)
        return lemmas



#inputText = "We're testing this right now, let's make sure it works!"
#preProcessObj = textPreProcessor(inputText)
#print(preProcessObj.expand_contractions(contraction_mapping=contraction_map))


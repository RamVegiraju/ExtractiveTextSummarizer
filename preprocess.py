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
    """
    A class for text preprocessing prior to text summarization and entity recognition.

    Attribute(s):
        text(str): The input text from user on front-end.
    """

    def __init__(self,text):
        self.text = text

    def expand_contractions(self, contraction_mapping=contraction_map):
        """Expand contractions"""
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
    
    def text_lowercase(self): 
        """Convert to lowercase"""
        return self.text.lower()
    
    def convert_number(self): 
        """Convert numbers to words."""
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
    
    def remove_punctuation(self): 
        """Remove punctuation."""
        translator = str.maketrans('', '', string.punctuation) 
        return self.text.translate(translator)
    
    def remove_whitespace(self): 
        """Remove whitespace."""
        return  " ".join(self.text.split())
    
    def word_lemmatizer(self):
        """Lemmatize words."""
        lemmatizer = WordNetLemmatizer()
        lem_text = [lemmatizer.lemmatize(i) for i in self.text]
        lemmas = ''.join(lem_text)
        return lemmas
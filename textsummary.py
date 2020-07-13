import pandas as pd 
import json
from json import loads
import numpy as np
import re
import six
import nltk
import ssl
import spacy
import string
import ast
import inflect
import heapq
import unicodedata
import matplotlib.pyplot as plt
from spacy import displacy
from collections import defaultdict
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from preprocess import expand_contractions,text_lowercase,convert_number,remove_punctuation,remove_whitespace,word_lemmatizer, word_tokenizer, sentence_tokenizer
stopwords = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')

"""
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
"""

file = open("contractions.txt", "r")
contents = file.read()
contraction_map = eval(contents)
file.close()
#print(type(contraction_map))

inputText = """Those Who Are Resilient Stay In The Game Longer
“On the mountains of truth you can never climb in vain: either you will reach a point higher up today, or you will be training your powers so that you will be able to climb higher tomorrow.” — Friedrich Nietzsche
Challenges and setbacks are not meant to defeat you, but promote you. However, I realise after many years of defeats, it can crush your spirit and it is easier to give up than risk further setbacks and disappointments. Have you experienced this before? To be honest, I don’t have the answers. I can’t tell you what the right course of action is; only you will know. However, it’s important not to be discouraged by failure when pursuing a goal or a dream, since failure itself means different things to different people. To a person with a Fixed Mindset failure is a blow to their self-esteem, yet to a person with a Growth Mindset, it’s an opportunity to improve and find new ways to overcome their obstacles. Same failure, yet different responses. Who is right and who is wrong? Neither. Each person has a different mindset that decides their outcome. Those who are resilient stay in the game longer and draw on their inner means to succeed.
I’ve coached many clients who gave up after many years toiling away at their respective goal or dream. It was at that point their biggest breakthrough came. Perhaps all those years of perseverance finally paid off. It was the 19th Century’s minister Henry Ward Beecher who once said: “One’s best success comes after their greatest disappointments.” No one knows what the future holds, so your only guide is whether you can endure repeated defeats and disappointments and still pursue your dream. Consider the advice from the American academic and psychologist Angela Duckworth who writes in Grit: The Power of Passion and Perseverance: “Many of us, it seems, quit what we start far too early and far too often. Even more than the effort a gritty person puts in on a single day, what matters is that they wake up the next day, and the next, ready to get on that treadmill and keep going.”
I know one thing for certain: don’t settle for less than what you’re capable of, but strive for something bigger. Some of you reading this might identify with this message because it resonates with you on a deeper level. For others, at the end of their tether the message might be nothing more than a trivial pep talk. What I wish to convey irrespective of where you are in your journey is: NEVER settle for less. If you settle for less, you will receive less than you deserve and convince yourself you are justified to receive it.
“Two people on a precipice over Yosemite Valley” by Nathan Shipps on Unsplash
Develop A Powerful Vision Of What You Want
“Your problem is to bridge the gap which exists between where you are now and the goal you intend to reach.” — Earl Nightingale
I recall a passage my father often used growing up in 1990s: “Don’t tell me your problems unless you’ve spent weeks trying to solve them yourself.” That advice has echoed in my mind for decades and became my motivator. Don’t leave it to other people or outside circumstances to motivate you because you will be let down every time. It must come from within you. Gnaw away at your problems until you solve them or find a solution. Problems are not stop signs, they are advising you that more work is required to overcome them. Most times, problems help you gain a skill or develop the resources to succeed later. So embrace your challenges and develop the grit to push past them instead of retreat in resignation. Where are you settling in your life right now? Could you be you playing for bigger stakes than you are? Are you willing to play bigger even if it means repeated failures and setbacks? You should ask yourself these questions to decide whether you’re willing to put yourself on the line or settle for less. And that’s fine if you’re content to receive less, as long as you’re not regretful later.
If you have not achieved the success you deserve and are considering giving up, will you regret it in a few years or decades from now? Only you can answer that, but you should carve out time to discover your motivation for pursuing your goals. It’s a fact, if you don’t know what you want you’ll get what life hands you and it may not be in your best interest, affirms author Larry Weidel: “Winners know that if you don’t figure out what you want, you’ll get whatever life hands you.” The key is to develop a powerful vision of what you want and hold that image in your mind. Nurture it daily and give it life by taking purposeful action towards it.
Vision + desire + dedication + patience + daily action leads to astonishing success. Are you willing to commit to this way of life or jump ship at the first sign of failure? I’m amused when I read questions written by millennials on Quora who ask how they can become rich and famous or the next Elon Musk. Success is a fickle and long game with highs and lows. Similarly, there are no assurances even if you’re an overnight sensation, to sustain it for long, particularly if you don’t have the mental and emotional means to endure it. This means you must rely on the one true constant in your favour: your personal development. The more you grow, the more you gain in terms of financial resources, status, success — simple. If you leave it to outside conditions to dictate your circumstances, you are rolling the dice on your future.
So become intentional on what you want out of life. Commit to it. Nurture your dreams. Focus on your development and if you want to give up, know what’s involved before you take the plunge. Because I assure you, someone out there right now is working harder than you, reading more books, sleeping less and sacrificing all they have to realise their dreams and it may contest with yours. Don’t leave your dreams to chance."""

#print(len(inputText))
#print('---------------')
#print("that is the length of the input")

#summarizer function
def freq_calculator(text):
    #sample_json = json.dumps(sample_json)
    #sample_json = json.loads(sample_json)
    #text = sample_json["Input"]

    #Applying preprocessing functions
    #text = expand_contractions(text, contraction_mapping=contraction_map)
    text = text_lowercase(text)
    text = convert_number(text)
    #text = remove_punctuation(text)
    text = remove_whitespace(text)

    #storing word frequencies and removing stop words
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    #calculating average value of each word
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    #calculating sentence value 
    sentence_scores = {}
    sentence_list = sent_tokenize(text)
    number_sentences = len(sentence_list)
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]
    if (number_sentences<=5):
        summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        summary = summary.capitalize()
    elif (number_sentences==1):
        summary_sentences = text
        summary = summary.capitalize()
    else:
        summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)
        summary = ' '.join(summary_sentences)
        summary = summary.capitalize()
    return summary

#sample_obj = {"Input": "Those Who Are Resilient Stay In The Game Longer."}
#print(freq_calculator(sample_obj))

#find entities in summary
def find_entities(text):
    text = nlp(text)
    named_entities = {}
    if text.ents:
        for ent in text.ents:
            named_entities[ent] = ent.label_
    else:
        pass
    return named_entities

#output entities in summary
def show_entities(text):
    #summaryText = freq_calculator(text)
    new_dict = {}
    #new_dict = defaultdict(set)
    named_entities = find_entities(text)
    for key, value in named_entities.items(): 
        if value in new_dict: 
            new_dict[value].append(key) 
        else: 
            new_dict[value]=[key] 
    entity_table = pd.DataFrame(new_dict.items())
    return entity_table

#entities_summary = show_entities(inputText)
#print(entities_summary)

def mainFunc(text):
    summary = freq_calculator(text)
    entity_dict = show_entities(summary)
    return summary, entity_dict

#print(mainFunc(inputText))


#Pushing to git
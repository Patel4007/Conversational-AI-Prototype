import spacy
import nltk
import sys
import requests
import re
import string
from heapq import nlargest
from newspaper import Article
from nltk.tokenize.punkt import PunktSentenceTokenizer
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import math
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

nltk.download('words')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import words
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer
from string import punctuation
from nn_model import model, words, labels 

# Function to sort the list
def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))
    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                list_index[i], list_index[j] = list_index[j], list_index[i]
    return list_index

# Preprocess text function
def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text

# Function to create list of probabilities
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

# Function to generate response
def response(user_input):
    user_input = '+'.join(user_input.split())
    
    article = Article("https://www.google.com/search?q=" + user_input)
    article.download()
    article.parse()
    article.nlp()
    corpus = article.text
    text = corpus
    sent_tokenizer = PunktSentenceTokenizer(text)
    sentence_list = sent_tokenizer.tokenize(text)  # nltk.sent_tokenize(text)

    user_input = user_input.lower()
    response = ''
    
    count_matrix = CountVectorizer(stop_words=["are", "all", "in", "and"]).fit_transform(sentence_list)  
    similarity_matrix = cosine_similarity(count_matrix[-1], count_matrix)
    sparse_array = similarity_matrix.flatten()

    
    index = index_sort(sparse_array)
    index = index[1:]

    response_flag = False
    best_match = 'I did not understand the query'

    for i in range(len(index)):
        if sparse_array[index[i]] > 0.0:
            response = ' '.join([sentence_list[index[i]]])
            best_match = response
            response_flag = True
            break

    if not response_flag:
        best_match = 'I could not find any relevant information.'

    return best_match

# Main method
if __name__ == "__main__":

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))

        else:
            responses = response(inp)
            
        print(responses)

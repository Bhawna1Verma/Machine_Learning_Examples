# simple chatbot using python: sklearn and nltk
"""
Empahasis on sklearn and nltk
 - used for text processing, tokenization and Lemmetization
Hard coding for greetings  part (not using AI for greetings and response) to keep it simple
"""

import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

import nltk # natural language toolkit
from nltk.stem import WordNetLemmatizer
nltk.download('popular',quiet=True) # for downloading packages

nltk.download('punkt')
nltk.download('wordnet')

# Reading the corpus
with open('chatbot.txt','r',encoding = 'utf8', errors= 'ignore') as fin:
    raw=fin.read().lower()

#Tokenization
sent_tokens = nltk.sent_tokenize(raw) # converts to list of sentences
word_tokens = nltk.word_tokenize(raw) # converts to list of words

#Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return[lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword matching 
GREETING_INPUTS = ('hello','hi','greetings','sup',"what's up",'hey')
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    # returns a greeting response for users greeting input
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        
# main part: generating the response based on cosine similarity
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize,stop_words='english') # converts the sentence tokes to vectors 
    tfidf = TfidVec.fit_transform(sent_tokens)   # estimated the importance of dat in numerical form 
    vals = cosine_similarity(tfidf[-1],tfidf)  # finding similarity between user responce (tfidf[-1]) and all the sentences(tfidf). 
    idx = vals.argsort()[0][-2]   # finding idx of the best responce since -1 is the user responce therefore going to -2
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2] # finding the second most similar to users input
    if(req_tfidf==0):
        robo_response = robo_response + 'I am sorry! I donot understand you'
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

# coninuously ask for users response untill he says bye
flag = True
print('ROBO: My name is robo. I will answer your queries about chatbots. If you want to exit, type Bye!')

while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response!='bye'):
        if(user_response=="thanks" or user_response == "thank you"):
            flag=False
            print('ROBO: You are welcome..')
        else:
            if(greeting(user_response)!=None):
                print('ROBO'+greeting(user_response))
            else:                                                  # main loop that takes the data and vectorize and removes after responding
                print('ROBO: ',end='')
                print(response(user_response))
                sent_tokens.remove(user_response)        

    else:
        flag = False
        print('ROBO: Bye! take care..')    

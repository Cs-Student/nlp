# imports
from collections import Counter
import os
import spacy
import nltk
from natsort import natsorted
import re
from nltk.tokenize import sent_tokenize
from num2words import num2words


def print_details(text,word):
    try:
        words = re.findall(r'\w+', text)
        number_of_sentences = sent_tokenize(text)
        print("Number of Words are {} and number of Sentences are {}".format(str(len(words)),str(len(number_of_sentences))))
        count_of_cons = len(re.findall(r"\b[^aeiouAEIOU\s][a-zA-Z]*\b", text))
        count_of_vowel = len(re.findall(r"\b[aeiouAEIOU][a-zA-Z]*\b", text))
        print("Number of words with first alphabet consonant are {} \n"
              "and number of words with first alphabet vowel are {}".
              format(str(count_of_cons),str(count_of_vowel)))
        list_of_email = re.findall(r"\S+@\S+", text)
        print("email id present in the text are:")
        print(list_of_email)
        cntStart = []
        cntEnd = []
        nlp = spacy.load('en_core_web_sm')
        tokens = nlp(text)
        for i in tokens.sents:
            i = i.string.strip()
            ar = i.split()
            if ar[0] == word:
                cntStart.append(i)
            if word in ar[len(ar) - 1]:
                cntEnd.append(i)
        print("{} sentences with word at starting : ".format(str(len(cntStart))))
        print(cntStart)
        print("{} sentences with word at ending : ".format(str(len(cntEnd))))
        print(cntEnd)
        wc = Counter(words)
        wordCount = 0
        if word in wc:
            wordCount = wc[word]
        list_of_statement = []
        for i in number_of_sentences:
            if word in i:
                list_of_statement.append(i)
        print("{} count of word {} : ".format(str(wordCount),word))
        print("statements that contains word {}".format(word))
        print(list_of_statement)
        list_of_questions = []
        for i in number_of_sentences:
            if '?' in i:
                list_of_questions.append(i)
        print("questions present are")
        print(list_of_questions)
        date = re.findall(r"Date: [^\n]*", text)[0]
        min = date[26:28]
        sec = date[29:31]
        print("minutes: {} and seconds{} ".format(min,sec))
        abr = re.findall(r"\b(?:[A-Z.]+){2,}\b", text)
        print("abbreviations present")
        print(abr)
    except:
        print("unexpected input")

# getting the filename,folder and word and extracting the text out of the file

def get_info(fol,filename,word):
    path = "/home/lucky/data/iiitd/dead/nlp/asm01/data/"
    # filepath = "/home/lucky/Downloads/Dev_set.txt"
    dir = [d[0] for d in natsorted(os.walk(path))]
    if fol not in [1, 2]:
        return
    location = dir[fol]
    filepath = location+"/"+filename
    op = open( filepath,'r',encoding="utf8", errors='ignore')
    text = op.read().strip()
    idx = text.find('\n\n')
    text = text[idx+2:]
    text = text.replace(">--", " ")
    text = text.replace(">", " ")
    text = re.sub('\d+',lambda num : num2words(num.group()),text)
    print_details(text,word)

# interactive menu for input
while True:
    fol = int(input(" Enter 1 for rec.motorcycles and 2 for sci.med "))
    name = input(" Enter name of the file ")
    word = input(" Enter the word ")
    get_info(fol,name,word)

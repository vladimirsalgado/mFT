#
#this code is to run FastText and to find the best parameters for introduced dataset
#runft_v dataname trainfilename language

import subprocess
import sys
import re
import nltk
import random
from itertools import product
import multiprocessing
import os
from nltk.stem import *
#from nltk.stem.snowball import SnowballStemmer

dataset = sys.argv[1]
trainfile = sys.argv[2]
lang = sys.argv[3]
outputfile = open(dataset + ".log","w")

#Cut the train file in a 70 train, 30 validation proportion.
def CreateValidaFile(inputfile, percentvalid):
    p = subprocess.Popen("wc -l " + inputfile, stdout=subprocess.PIPE, shell=True)
    p.wait()
    wcoutput = p.stdout.read().decode()
    capture = re.split(' ', wcoutput)
    #print(capture)
    lines = int(capture[0])

    numbervalidation = int(lines*percentvalid)
    #print(str(numbervalidation))
    numbertraining = lines-numbervalidation
    #print(str(numbertraining))

    p = subprocess.Popen("shuf " + inputfile + " -o " + inputfile + ".shuf", shell=True)
    p.wait()

    p = subprocess.Popen("head -n " + str(numbervalidation) + " " + inputfile + ".shuf > " + inputfile + ".valid", shell=True)
    p.wait()
    p = subprocess.Popen("tail -n " + str(numbertraining) + " " + inputfile + ".shuf > " + inputfile + ".train", shell=True)
    p.wait()
    print("Validation File Created.")


def Punctuationize(args):
    datafile = args

    filename = open(datafile)
    stopfile = open(datafile + ".p","w")
    punctuation = ['.',',','!','?','"',"'","~", "/", "\\", ";", ":", "<", ">", "]", "[", "(", ")", "-", "=", "+", "|", "*", "$", "%", "^", "#", "@", "`", "&", "}", "{"]

    with filename as f:
        for line in f.readlines():
            sentence = [i for i in line if i not in punctuation]
            newline = ''.join(sentence)
            stopfile.write(newline)
    print("File without punctuation symbols was created.")

def Stopwordize(args):
    datafile = args

    filename = open(datafile)
    stopfile = open(datafile + ".s","w")

    from nltk.corpus import stopwords
    stop = set(stopwords.words('english'))

    with filename as f: 
        for line in f.readlines():
            sentence = [i for i in line.lower().split() if i not in stop]
            newline = ' '.join(sentence)
            stopfile.write(newline + '\n')
    print("File without stopwords was created.")

def Lematize(args):
    datafile = args

    filename = open(datafile)
    lemafile = open(datafile + ".l","w")
    
    from nltk.stem import WordNetLemmatizer
    wnl = WordNetLemmatizer()

    with filename as f: 
        for line in f.readlines():
            words = nltk.word_tokenize(line)
            newline = ""
            for x in words:
                lemaword = wnl.lemmatize(x)
                newline += " " + lemaword
            lemafile.write(newline + '\n')
    print("File lematized was created.")


def Stemeedize(args, lang):
    datafile = args

    filename = open(datafile)
    stemfile = open(datafile + ".t","w")

    if lang == 'EN': 
        stemmer = PorterStemmer()
    elif lang == 'SP':
        stemmer = SnowballStemmer("spanish")

    with filename as f: 
        for line in f.readlines():
            words = nltk.word_tokenize(line)
            newline = ""
            for x in words:
                stemword = stemmer.stem(x)
                newline += " " + stemword
            stemfile.write(newline + '\n')
    print("File Stemmed was created.")

#do shuffle to validation and train file
#add stemmer with Porter with English, snowball for others languages
#check for other languages

#add argument for select language

    
#Create validation file. 
porcentaje = .3
CreateValidaFile(trainfile,porcentaje)

Punctuationize(trainfile + ".train")
Stopwordize(trainfile + ".train")
Lematize(trainfile + ".train")
Stemeedize(trainfile + ".train", lang)

Stopwordize(trainfile + ".train.p")

Lematize(trainfile + ".train.p")
Lematize(trainfile + ".train.s")
Lematize(trainfile + ".train.t")

Lematize(trainfile + ".train.p.s")

Stemeedize(trainfile + ".train.p", lang)
Stemeedize(trainfile + ".train.s", lang)
Stemeedize(trainfile + ".train.p.s", lang)

Stemeedize(trainfile + ".train.p.l", lang)
Stemeedize(trainfile + ".train.s.l", lang)
Stemeedize(trainfile + ".train.p.s.l", lang)

#tlsp
#0000
#0001 p
#0010 s
#0011 sp
#0100 l 
#0110 ls 
#0101 lp
#0111 lsp
#1000 t
#1001 tp
#1010 ts 
#1011 tsp 
#1100 tl 
#1101 tlp 
#1110 tls 
#1111 tlsp 



print("Created files.")


#
#this code is to run FastText and to find the best parameters for introduced dataset
#Sintax: mFT dataname trainfile validfile testfile NumOfParametersCombination 
#

import subprocess
import sys
import os
import re
import nltk
import random
from itertools import product
import multiprocessing
import numpy as np
import json

dataset = sys.argv[1] #Name of dataset
trainfile = sys.argv[2] #Name of Train File
validfile = sys.argv[3] #Name of Validation File
testfile = sys.argv[4] #Name of Test File
SpaceParams = sys.argv[5] #Number of parameters combination

def runfasttext(args):
    Name, Train, Validation, Epoch, LR, WordNgrams, Ws, Dim = args
    
    p = subprocess.Popen("./fasttext supervised -thread 1 -input " + Train + " -output " + Name + ".model -epoch " + str(Epoch) + " -lr " + str(LR) + " -wordNgrams " + str(WordNgrams) + " -ws " + str(Ws) + " -dim " + str(Dim), shell=True)
    p.wait()
    p = subprocess.Popen("./fasttext test " + Name + ".model.bin " + Validation, stdout=subprocess.PIPE, shell=True)
    ftoutput = p.stdout.read().decode() 
    performance = re.split('\n|\t',ftoutput)
    try:
        
        P0 = float(performance[3])
        R0 = float(performance[5])
        if P0 != 0 or R0 != 0:
            F1 = 2*((P0*R0)/(P0+R0))
        else:
            F1 = 0 
            
    except IndexError as e:
        print("Error: " + str(e))
        return ([Train,-1,Epoch,LR,WordNgrams,Ws,Dim])
        
    #delete temp files    
    os.unlink(Name + ".model.bin")
    os.unlink(Name + ".model.vec")
    
    return([F1,Epoch,LR,WordNgrams,Ws,Dim,Train])


def param_generator():
    TRF = ["", ".p", ".s", ".l", ".t", ".p.s",".p.l", ".s.l", ".p.s.l", ".p.t", ".s.t", ".t.l", ".p.s.t", ".p.l.t", ".s.l.t", ".p.s.l.t"]
    EPOCHs = random.sample(range(5,50,5),5)
    WORDNGRAMSs = random.sample(range(1,6,1),5)
    LRs = random.sample((.1, .2, .3, .4, .5, .6, .7, .8, .9, 1),10)
    WS = random.sample(range(3,8,2),3)
    DIM = random.sample((3,10,30,100,300),5) 

    L = []

    for i,k,j,l,m,d in product(EPOCHs, LRs, WORDNGRAMSs, WS, DIM, TRF):
        Params = (dataset+str(i)+str(k)+str(j)+str(l)+str(m)+d,trainfile+d,validfile,str(i),str(k),str(j),str(l),str(m))
        L.append(Params)

    return(L)

#Run fasttext
L = param_generator()
np.random.shuffle(L)
pool = multiprocessing.Pool(20)
SCORES = pool.map(runfasttext,L[:int(SpaceParams)])
    
SCORES.sort(reverse=True)

with open(dataset + ".log","a") as outputfile:
    for item in SCORES:
        out = json.dumps({'F1':item[0],'Epoch':int(item[1]),'LR':float(item[2]),'WordNGrams':int(item[3]),'WS':int(item[4]),'Dim':int(item[5]),'DataFile':item[6]})
        print(out, file=outputfile)
outputfile.close

print('\n\nNumber of parameters combination: ' + SpaceParams)
BP = json.dumps({'F1':SCORES[0][0], 'Epoch':int(SCORES[0][1]), 'LR':float(SCORES[0][2]), 'WordNGrams':int(SCORES[0][3]), 'WS':int(SCORES[0][4]), 'Dim':int(SCORES[0][5]), 'DataFile':SCORES[0][6]})
print('Best parameters found: ' + BP)

#Run Fasttext with Test File with the best parameter combination.
LL = (dataset, SCORES[0][6], testfile, SCORES[0][1], SCORES[0][2], SCORES[0][3], SCORES[0][4], SCORES[0][5])
FINALSCORES = runfasttext(LL)
FS = json.dumps({'F1':FINALSCORES[0], 'Epoch':int(FINALSCORES[1]), 'LR':float(FINALSCORES[2]), 'WordNGrams':int(FINALSCORES[3]), 'WS':int(FINALSCORES[4]), 'Dim':int(FINALSCORES[5]), 'DataFile':FINALSCORES[6]})
print('Scores in test file:' + FS)




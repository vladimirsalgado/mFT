#This code was made to transform input format
#from mTC to FastText
#
#SINTAXIS: mtc2ft inputfilename outputfilename

import sys
import json
#import so

inputfile=open(sys.argv[1])
outputfile=open(sys.argv[2],"w")

with inputfile as f:
    for line in f.readlines():
        linejson = json.loads(line)
        #write a file in fasttext format
        outputfile.write("__label__" + linejson["klass"] + " " + linejson["text"]+'\n')


import json
import string
import os
 


def cleanText( text ):
    return (text.translate(str.maketrans('', '', string.punctuation)).lower())

def proccess_file( filename ):
    arq = open(filename, encoding="utf8", errors='ignore')

    treino = True
    i = 0
    arq = arq.readlines()
    tam = len(arq)
    lim = int( 0.6 * tam)
    curr_id = ""
    curr_txt = ""
    curr_subject = ""
    newElement = ""
    messages_list = []
    first = True

    for line in arq:

        if("Newsgroup:" in line):
            if(curr_id != ""):
                if( i <= lim):
                    print(curr_id+","+curr_subject+","+cleanText(curr_txt), end="\n")
            line = line.split(" ")
            curr_subject = line[1].strip('\n')
            first = False

        elif ("document_id" in line or "Document_id" in line):
            line = line.split(" ")
            curr_id = line[1].strip('\n')
            curr_txt = ""
        elif("In article" not in line and "From" not in line and "Subject" not in line and "archivename" not in line):
            line = line.strip("<>\t\n ")
            curr_txt += line
        i+=1


for file in os.listdir("data/"):
    #print(file)
    proccess_file("data/"+file)
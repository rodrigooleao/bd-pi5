from pyspark.sql import SparkSession
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
import mmh3
import json
import string
import os
from sklearn import metrics


TAM_SHINGLE = 10

def shingling_text(spark, text, tam):
    shingles = [text[i:i + tam] for i in range(len(text) - tam + 1)]

    hs = [mmh3.hash(item) for item in shingles]

    return hs

def cleanText( text ):
    return (text.translate(str.maketrans('', '', string.punctuation)).lower())

def proccess_file( filename ):
    arq = open(filename, encoding="utf8", errors='ignore')

    treino = open("treino.txt", "w")
    teste = open("teste.txt", "w")


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
                entry = curr_id+","+curr_subject+","+cleanText(curr_txt)
                if( i <= lim):
                    treino.write(entry+"\n")
                else:
                    teste.write(entry+"\n")
                    

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
    treino.close()
    teste.close()

spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.executor.memory", "16g").config("spark.driver.memory", "16g").getOrCreate()
#LIMPEZA DA BASE
print("###LIMPEZA DA BASE###")
for file in os.listdir("data/"):
    print(file)
    proccess_file("data/"+file)
#TREINO
print("###TREINO DO MODELO###")
arq = open("treino.txt").readlines()

items_list = []
for line in arq:
    line = line.split(",")
    idd = int(line[0])
    clazz = line[1]
    text = line[2]

    shingled = shingling_text(spark, text, 10)
    tam = sum(shingled)
    
    if(tam != 0):
        item = ( idd , clazz, Vectors.dense(shingled))
        items_list.append(item)


dfItems = spark.createDataFrame(items_list, ["id","class","features"])

mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=50)

model = mh.fit(dfItems)

transformed = model.transform(dfItems).repartition(8)
transformed.cache()

#TESTE
print("###TESTE DO MODELO###")
arq = open("teste.txt").readlines()

number_of_tests = len(arq)
hits = 0

actual = []
predicted = []
i = 0
for line in arq:
    line = line.split(",")
    idd = int(line[0])
    clazz = line[1]
    text = line[2]
    shingled = shingling_text(spark, text, 10)

    tam = sum(shingled)

    if( tam != 0):
        querie = Vectors.dense(shingled)

        print(i)
        i+=1
        computed = model.approxNearestNeighbors(transformed, querie, 15).groupBy("class").count().orderBy(col("count").desc()).take(1)[0][0]
        print("Real:", clazz)
        actual.append(clazz)
        print("Computed:", computed)
        predicted.append(computed)
        print("Iguais:", clazz == computed, end="\n\n")
        print("\n\n")


print("FINAL: ", hits, number_of_tests)

##AVALIAÇÃO DO MODELO
print("###AVALIAÇÃO DO MODELO###")


print(metrics.classification_report(actual, predicted, digits=3))









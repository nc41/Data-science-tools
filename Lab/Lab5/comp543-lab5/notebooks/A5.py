from pyspark import SparkContext
sc = SparkContext(master="local[4]")


import re
import numpy as np
import math
import heapq
import collections
import time

lengthOfdic = 20000

# load up all of the  26,754 documents in the corpus
corpus = sc.textFile("s3://risamyersbucket/A5/a5KDtrainingV2.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x: 'id=' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x: (
    x[x.index('id="') + 4: x.index('">\t')], x[x.index('>\t') + 8: x.index('</doc>')]))

# keyAndText = validLines.map(lambda x: (
# x[x.index('id="') + 4: x.index('">\t      ') - 1], x[x.index('">\t
# ') + 2: x.index('</doc>')]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x: (
    str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2",
# 3423423), etc.
allCounts = allWords.reduceByKey(lambda a, b: a + b)

# and get the top 20,000 words in a local array
# each entry is a ("word1", count) pair
topWords = allCounts.top(lengthOfdic, lambda x: x[1])
# print(topWords)
# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(lengthOfdic))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords in the lambda that you supply
dictionary = twentyK.map(lambda x: (topWords[x][0], x))


# finally, print out some of the dictionary, just for debugging
# print(dictionary.collect())
# keyAndListOfWords.collect()
# keyAndText.collect()
docNum = keyAndText.count()
# test = sc.parallelize([(i, [1,2,3]) for i in range(9)])
# test = test.map(lambda x : (x[0], np.array(x[1])))
# test.values().sampleStdev()
# keyAndListOfWords.lookup("0NCT02586545")

"""
Task1
"""

target = ["applicant", "and", "attack", "protein", "car"]

# t1 = sc.parallelize([(i ,dictionary.lookup(i)[0]) if
# len(dictionary.lookup(i)) == 1 else (i, -1) for i in target])

print("Task1:", [(i, dictionary.lookup(i)[0]) if len(
    dictionary.lookup(i)) == 1 else (i, -1) for i in target])

"""
Task2
"""


def buildDic(wordIdx):
    res = np.zeros(lengthOfdic)
    for i in wordIdx:
        res[i] += 1

    return res


def listdivide(wordIdx):
    to = np.sum(wordIdx)
    return np.divide(wordIdx, to)


def listToOne(wordIdx):
    return np.where(wordIdx > 0, 1, 0)


# get the word idx in Dictionary and the total number each doc has.
# (docID, idx(wordPos in dictionary))
start_time = time.time()
wordPosInDict = dictionary.join(keyAndListOfWords.flatMap(
    lambda x: ((j, x[0]) for j in x[1]))).map(lambda x: (x[1][1], x[1][0]))

# (docId, (pos1, pos2, .....))
docWordPos = wordPosInDict.groupByKey()
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
PosArrayDoc = docWordPos.map(lambda x: (x[0], buildDic(x[1])))
print("--- %s seconds ---" % (time.time() - start_time))

# PosArrayDoc.top(1)

start_time = time.time()
TF = PosArrayDoc.map(lambda x: (x[0], listdivide(x[1])))
# TF.top(1)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
docOne = PosArrayDoc.map(lambda x: (x[0], listToOne(x[1])))
# docOne.top(1)

# # the number of docs of each word
Counts = docOne.reduce(lambda a, b: ("", a[1] + b[1]))[1]
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
IDF = np.log(np.divide(np.full(lengthOfdic, docNum), Counts))

TFIDF = TF.map(lambda x: (x[0], x[1] * IDF))
print("--- %s seconds ---" % (time.time() - start_time))
# TFIDF.top(1)

# normalize the TFIDF
start_time = time.time()
mean = TFIDF.values().mean()
std = TFIDF.values().sampleStdev()
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
std[std == 0] = 1
TFIDF_Norm = TFIDF.map(lambda x: (x[0], (x[1] - mean) / std))
TrainLabel = TFIDF_Norm.map(lambda x: (
    x[0], (x[1], 1 if x[0][0] == '1' else 0)))
print("--- %s seconds ---" % (time.time() - start_time))
# TrainLabel.top(1)


def LogTraining(Loss_Diff, lr, regulization, epoch, verbose=True):
    #     w = 2*np.random.sample((lengthOfdic,)) - 1
    w = np.zeros((lengthOfdic,))

    pre_Loss = 99999999

    for i in range(epoch):
        start_time = time.time()
        H = TrainLabel.map(lambda x: (
            x[0], (x[1][0], x[1][1], 1.0 / (1.0 + np.exp(-np.dot(x[1][0], w)))))
#         get rid of the 1 into our data because log(0) does not exist.
        H=H.map(lambda x: (x[0], (x[1][0], x[1][1],
                                    (1 - 1e-9 if x[1][2] == 1 else x[1][2]))))
        c=H.map(lambda x: (
            x[0], (-x[1][1] * math.log(x[1][2]) - (1 - x[1][1]) * math.log(1 - x[1][2]))))
#         print(c.reduce(lambda a, b: ('', a[1] + b[1]))[1])
        l=(1.0 / docNum) * (c.reduce(lambda a, b: ('', a[1] + b[1]))[1]) + (
            regulization / (2.0 * docNum)) * np.sum(np.square(w))

#         return gradient

        gradient=H.map(lambda x: (
            x[0], (x[1][2] - x[1][1]) * x[1][0])).reduce(lambda a, b: ('', a[1] + b[1]))[1]

        update=gradient / docNum + regulization * w / docNum
        w -= lr * update

        if l <= pre_Loss:
            lr *= 1.1
        else:
            lr *= 0.5

        if verbose:
            print('Epoch:{0}; Loss:{1}'.format(i, l))

        print("--- %s seconds ---" % (time.time() - start_time))

        if abs(pre_Loss - l) < Loss_Diff:
            break

        pre_Loss=l

    return w


weight=LogTraining(0.0001, 2, 0.5, 300, True)

# test result
K=50


def getTopWeightedWord(K=50):
    TopKWords=weight.argsort()[-K:][::-1]
    dic=dictionary.collect()
    res=[]
    for i in TopKWords:
        print("WeightW:", dic[i][0])
        res.append(dic[i][0])

    # res_rdd = sc.parallelize(res)
    # res_rdd.coalesce(1).saveAsTextFile("output_task_2")


getTopWeightedWord()

'''
task 3
'''
# load up all of the documents in the corpus
corpusT=sc.textFile("s3://risamyersbucket/A5/a5KDtestingV2.txt")

# each entry in validLines will be a line from the text file
validLinesT=corpusT.filter(lambda x: 'id=' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndTextT=validLinesT.map(lambda x: (
    x[x.index('id="') + 4: x.index('">\t')], x[x.index('>\t') + 8: x.index('</doc>')]))

# keyAndText = validLines.map(lambda x: (
# x[x.index('id="') + 4: x.index('">\t      ') - 1], x[x.index('">\t
# ') + 2: x.index('</doc>')]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex=re.compile('[^a-zA-Z]')
keyAndListOfWordsT=keyAndTextT.map(lambda x: (
    str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", docID) ("word2", docID)...
allWordsT=keyAndListOfWordsT.flatMap(lambda x: ((j, x[0]) for j in x[1]))

wordPosInDictT=dictionary.join(allWordsT)

w=wordPosInDictT.map(lambda x: (x[1][1], x[1][0]))

docWordPosT=w.groupByKey()
# docWordPosT.top(1)
PosArrayDocT=docWordPosT.map(lambda x: (x[0], buildDic(x[1])))
# PosArrayDocT.take(2)
TFT=PosArrayDocT.map(lambda x: (x[0], listdivide(x[1])))

docOneT=PosArrayDocT.map(lambda x: (x[0], listToOne(x[1])))
# docOneT.take(1)

# # # the number of docs of each word
CountsT=docOneT.reduce(lambda a, b: ("", a[1] + b[1]))[1]

TFIDFT=TFT.map(lambda x: (x[0], x[1] * IDF))
Norm_TFIDFT=TFIDFT.map(lambda x: (x[0], (x[1] - mean) / std))

# we get the weight with label here
Label_Z=Norm_TFIDFT.map(lambda x: (
    x[0], 1 if x[0][0] == '1' else 0, np.dot(x[1], weight)))


# predict by the weight.
Threshold=6
Prediction=Label_Z.map(lambda x: (x[0], x[1], 1 if x[2] > Threshold else 0))

TP=Prediction.map(lambda x: (
    '', 1 if x[1] == 1 and x[2] == 1 else 0)).values().sum()
FP=Prediction.map(lambda x: (
    '', 1 if x[1] == 0 and x[2] == 1 else 0)).values().sum()
TN=Prediction.map(lambda x: (
    '', 1 if x[1] == 0 and x[2] == 0 else 0)).values().sum()
FN=Prediction.map(lambda x: (
    '', 1 if x[1] == 1 and x[2] == 0 else 0)).values().sum()

F1=float(2 * TP) / float((2 * TP + FN + FP))

rate=float(Prediction.map(lambda x: (
    x[0], 1 if(x[1] == x[2]) else 0)).values().sum())
rate /= float(Prediction.count())

print("F1:{}".format(F1))

print(TP, FP, TN, FN, F1, rate)
FP_exp=Prediction.map(lambda x: (
    x[0], 1 if x[1] == 0 and x[2] == 1 else 0)).filter(lambda x: x[1] == 1).collect()
print(FP_exp[:3])

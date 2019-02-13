from pyspark import SparkContext
sc = SparkContext(master="local[4]")

import re
import numpy as np
import math
import heapq
import collections

# load up all of the  26,754 documents in the corpus
corpus = sc.textFile("s3://risamyersbucket/A4/pubmed.txt")

# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x: 'id=' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x: (
    x[x.index('id=') + 3: x.index('> ')], x[x.index('> ') + 2:]))

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
topWords = allCounts.top(20000, lambda x: x[1])

# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords in the lambda that you supply
dictionary = twentyK.map(lambda x: (topWords[x][0], x))


# finally, print out some of the dictionary, just for debugging
# dictionary.collect()

def tf(tokens):
    d = {}
    for word in tokens:
        if not word in d:
            d[word] = 1
        else:
            d[word] += 1
    for word in d:
        d[word] = float(d[word]) / len(tokens)
    return d


def IDF(RDD):
    N = RDD.count()
    uniqueTokens = RDD.map(lambda x: list(set(x[1])))
    tokenSumPairTuple = uniqueTokens.flatMap(lambda x: x).map(
        lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
    return (tokenSumPairTuple.map(lambda x: (x[0], math.log10(float(N) / x[1]))))


def distance(t1, t2):

    dst = np.sqrt(np.sum((t1 - t2)**2))
    return dst


def computeTFIDF(dic, idfs):
    dicWord = keyAndListOfWords.filter(lambda x: x[0] == dic)
    dicWord = dicWord.flatMap(lambda x: ((j, 1) for j in x[1]))
    wordInDic = dicWord.join(dictionary).map(lambda x: x[0])
    wordInDic = sc.parallelize(list(("", wordInDic.collect())))
    tfs = wordInDic.map(lambda x:  tf(x))
#     idfs = IDF(keyAndListOfWords)
    v = sc.parallelize(list(tfs.collect()[1].items()))
    tfidf = v.join(idfs)
    tfidf = tfidf.map(lambda x: (x[0], x[1][0] * x[1][1]))
    ans = dictionary.leftOuterJoin(tfidf).sortBy(lambda x: x[1][0])
    ans = ans.map(lambda x: x[1][1] if x[1][1] else 0)
    return ans


def computeTFIDF_all(idfs):
    doc = keyAndListOfWords.map(lambda x: x[0]).collect()
    dicWord = keyAndListOfWords.filter(lambda x: x[0] in doc[:400])
    tfs = dicWord.map(lambda x:  (x[0], tf(x[1])))
    return tfs


def kNN(k, tfidf, idfs):
    # doc = keyAndListOfWords.map(lambda x: x[0]).collect()
    # heap = []
    # target = tfidf.collect()
    # for d in doc:
    #     t = computeTFIDF(d, idfs)
    #     dst = distance(np.array(target), np.array(t.collect()))
    #     heapq.heappush(heap, (-dst, d))
    #     print(dst)
    #     if len(heap) > k:
    #         heapq.heappop(heap)

    # dic = collections.Counter()
    # for ele in heap:
    #     cata = ele[1].split('/')[0]
    #     dic[cata] += 1

    return dic.most_common(1)


def predictLabel(k, s):
    ss = s.split()
    Text = sc.parallelize(list([ss]))

    tfs = Text.map(lambda x:  tf(x))

    idfs = IDF(keyAndListOfWords)
    v = sc.parallelize(list(tfs.collect()[0].items()))
    tfidf = v.join(idfs)
    tfidf = tfidf.map(lambda x: (x[0], x[1][0] * x[1][1]))
    tfidf = dictionary.leftOuterJoin(tfidf).sortBy(lambda x: x[1][0])
    tfidf = tfidf.map(lambda x: x[1][1] if x[1][1] else 0)

    return kNN(k, tfidf, idfs)


print(predictLabel(10, 'Simulation technology for health care professional skills training and assessment.  Changes in medical practice that limit instruction time and patient availability, the expanding options for diagnosis and management, and advances in technology are contributing to greater use of simulation technology in medical education. Four areas of high-technology simulations currently being used are laparoscopic techniques, which provide surgeons with an opportunity to enhance their motor skills without risk to patients; a cardiovascular disease simulator, which can be used to simulate cardiac conditions; multimedia computer systems, which includes patient-centered, case-based programs that constitute a generalist curriculum in cardiology; and anesthesia simulators, which have controlled responses that vary according to numerous possible scenarios. Some benefits of simulation technology include improvements in certain surgical technical skills, in cardiovascular examination skills, and in acquisition and retention of knowledge compared with traditional lectures. These systems help to address the problem of poor skills training and proficiency and may provide a method for physicians to become self-directed lifelong learners.'))

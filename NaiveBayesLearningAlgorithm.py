import os
import re
from NaiveBaiseModel import NaiveBaiseModel
from pprint import pprint

def getClasses(path):
    trainClasses = next(os.walk(path))[1]
    return trainClasses


def tokenize(trText):
    # tokenList = re.split('\W+', " ".join(trText.lower()))
    match = re.findall('([a-zA-ZäöüÄÖÜß]{2,})', trText.lower())
    if match is None:
        return []

    return [x for x in match if x is not None]


def getUniqWords(words):
    uniqWords = {}
    for word in words:
        uniqWords[word] = uniqWords.get(word, 0) + 1
    return uniqWords


class NaiveBayesLearningAlgorithm(object):
    def __init__(self, dataPath, trainPath):
        self.dataPath = dataPath
        self.trainPath = trainPath
        self.calculateValues()

    def calculateValues(self):
        # tr = training
        absPath = os.path.abspath(self.dataPath)
        classes = getClasses(absPath)
        trainAbsPaths = [os.path.join(absPath, trCls, self.trainPath) for trCls in classes]
        docs = {}
        words = {}
        wordAmountInCls = {}
        docsCount = {}
        summaryNumberOfWords = 0
        summaryCountUniqWords = 0
        # def addWordsToClass(str, cls):

        for trCls in classes:
            trPath = os.path.join(absPath, trCls, self.trainPath)
            docNames = os.listdir(trPath)
            docsCount[trCls] = len(docNames)
            wordsForClass = []
            for name in docNames:
                f = open(os.path.join(trPath, name), 'r', encoding='utf-8')
                for line in f:
                    if not line:
                        continue
                    wordsForClass = wordsForClass + tokenize(line)
            words[trCls] = getUniqWords(wordsForClass)
            summaryCountUniqWords = summaryCountUniqWords + len(words[trCls])
            summaryNumberOfWords = summaryNumberOfWords + len(wordsForClass)
            wordAmountInCls[trCls] = len(wordsForClass)
        self.summaryNumberOfWords = summaryNumberOfWords
        self.words = words
        self.m = NaiveBaiseModel(wordAmountInCls, docsCount, words, summaryCountUniqWords)

    def classify(self, str):
        classes = list(self.words.keys())
        probs = {}
        for trCls in classes:
            probs[trCls] = self.calculateProbability(trCls, str)
            # returnmax([self.calculateProbability(trCls) for trCls inclasses])
        # pprint(probs)
        return max(probs, key = probs.get)

    def calculateProbability(self, cls, str):
        probs = self.m.calculatePriorProbability(cls)
        for word in tokenize(str):
            probs = probs + self.m.wordLogProbability(cls, word)
        return probs

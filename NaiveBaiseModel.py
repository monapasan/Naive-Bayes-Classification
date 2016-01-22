from math import log
from functools import reduce

def tokenize(trText):
    # tokenList = re.split('\W+', " ".join(trText.lower()))
    match = re.findall('([a-zA-ZäöüÄÖÜß]{2,})', trText.lower())
    if match is None:
        return []

    return [x for x in match if x is not None]


# length - length of class dicitionary
# docCount - number of Documents by class
# summaryNumberOfDocs
# wordCount - dictionary of all classes, that we have
# countUniqWords - Cmount of uniq word
class NaiveBaiseModel(object):
    def __init__(self, length, docCount, wordCount, countUniqWords):
        self.length = length
        self.docCount = docCount
        self.wordCount = wordCount
        self.countUniqWords = countUniqWords

    def wordLogProbability(self, trCls, word):
        # uniqWordsCount = reduce(lambda x, y: x + y, self.wordCount.values())
        uniqWordsCount = self.countUniqWords
        # print(uniqWordsCount)
        return log((self.wordCount[trCls].get(word, 0) + 1.0) / (self.length[trCls] + uniqWordsCount))

    def calculatePriorProbability(self, trCls):
        summaryNumberOfDocs = reduce(lambda x, y: x + y, self.docCount.values())
        return log(self.docCount[trCls] / summaryNumberOfDocs)

    def classify(self, str):
        print(wordLogProbability(str))
        print(calculatePriorProbability(str))

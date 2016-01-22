import os
from NaiveBayesLearningAlgorithm import NaiveBayesLearningAlgorithm
from pprint import pprint
from Tester import Tester
import sys
import codecs
import re
# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
# sys.stdout.encoding = 'utf-8'
def tokenize(trText):
    # tokenList = re.split('\W+', " ".join(trText.lower()))
    match = re.findall('([a-zA-ZäöüÄÖÜß]{2,})', trText.lower())
    if match is None:
        return []

    return [x for x in match if x is not None]

standardPath = "data"
trainPath = "train"
testPath = "test"

trainData = NaiveBayesLearningAlgorithm(standardPath, trainPath)
myTester = Tester(standardPath, testPath, trainData)
# print(myTester.trainAbsPaths)
#
# datatest = os.path.join(os.path.abspath(standardPath), 'sport', testPath, "s080.txt")
# # print(tokenize())
# print(trainData.classify(open(datatest).read()))
# NaiveBayesModel.train()
# pprint(str(trainData))
# pprint(trainData)
# for x in trainData:
#     for y in trainData[x]:
#         print(x, y.encode('utf8'))

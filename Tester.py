from NaiveBayesLearningAlgorithm import getClasses
import os


class Tester(object):
    def __init__(self, trainPath, testPath, trainer):
        self.trainPath = trainPath
        self.testPath = testPath
        self.trainer = trainer
        self.read()

    def read(self):
        absPath = os.path.abspath(self.trainPath)
        classes = getClasses(absPath)
        self.trainAbsPaths = [os.path.join(absPath, trCls, self.trainPath) for trCls in classes]
        results = {}
        for trCls in classes:
            trPath = os.path.join(absPath, trCls, self.testPath)
            docNames = os.listdir(trPath)
            print("\n", trCls, "\n")
            for name in docNames:
                doc = open(os.path.join(trPath, name), 'r', encoding='utf-8').read()
                results[name] = self.trainer.classify(doc)
                print(name, " ==> ", results[name])

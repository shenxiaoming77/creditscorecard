#coding=utf-8

import  numpy as np
import  pandas as pd

from  settings import  *

from  util.featureEngineering_functions  import *

class TrainDataGenerator:



    def loadFeatures(self, file):
        lines = loadFeatures(file)
        return  lines


if __name__ == '__main__':
    generator = TrainDataGenerator()
    features = generator.loadFeatures('features')
    print(features)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle

def updateClass():
    # find all saved pickles
    direct = glob.glob("./pickles/*")
    pickles = [glob.glob(file+"/*.pickle") for file in direct]
    for file in pickles:
        # open file and update functions in classes
        f = open(file,"rb")
        obj = pickle.load(f)
        f.close()
        # dump pickles back
        f = open(file,"wb")
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

def readClass(name):
    # find all saved pickles
    file = glob.glob("./pickles/"+name)
    pickles = glob.glob(file[0]+"/*.pickle")
    f = open(pickles[0],"rb")
    obj = pickle.load(f)
    f.close()
    return obj
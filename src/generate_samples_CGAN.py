#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from obj.CGAN import CGAN

def plotSamples(namePickle,nameFile):    
    dim = pd.read_csv("./pickles/"+namePickle+"/log.csv")["latent_dimensions"][0]
    # remove dropout for testing
    model = CGAN(dim, drop_rate=0)
    model.load_weights("./pickles/"+namePickle+"/cgan")
    collection = []
    while len(collection) < 100:
        samples = model.generate()
        collection.extend([samples[i,:,:,0] for i in tf.where(tf.nn.sigmoid(model.discriminate(samples))>0.5)[:,0]])
    collection = collection[:100]
    plotSamples_CGAN(collection, nameFile)

def plotSamples_CGAN(obj, name, nrows = 10, ncols = 10):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    i = 0
    for row in ax:
        for col in row:
            col.imshow(obj[i],cmap='Greys_r')
            col.axis('off')
            i += 1
    fig.savefig(os.path.abspath(os.path.dirname(os.getcwd()))+"/img/"+name+".png", dpi=400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test",
                        help="file name of output png <default: 'test'>")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-p', '--pickle', type=str,
                               help="name of directory where CGAN weights are stored",
                               required=True)
    args = parser.parse_args()
    plotSamples(args.pickle,args.out)
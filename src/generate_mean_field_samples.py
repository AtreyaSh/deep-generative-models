#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt
from aux.updateClass import readClass

def plotSamples(namePickle,nameFile,dim,number_samples = 100):
    dbn = readClass(namePickle)
    samples = dbn.generate_visible_downwards(number_samples = number_samples)
    plotSamples_DBN(samples, nameFile, dim)

def plotSamples_DBN(obj, name, dim, nrows = 10, ncols = 10):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    i = 0
    for row in ax:
        for col in row:
            col.imshow(tf.reshape(obj[i],(dim,dim)),cmap='Greys_r')
            col.axis('off')
            i += 1
    fig.savefig(os.path.abspath(os.path.dirname(os.getcwd()))+"/img/"+name+".png", dpi=400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test",
                        help="file name of output png, defaults to 'test'")
    parser.add_argument("--dim", type=int, default=28,
                        help="square dimensions on which to remap images, defaults to 28")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-p', '--pickle', type=str, 
                               help="name of directory where dbn.pickle is stored",
                               required=True)
    args = parser.parse_args()
    plotSamples(args.pickle,args.out,args.dim)
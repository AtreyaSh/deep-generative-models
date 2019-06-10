#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt
from obj.DBN import DBN
from aux.updateClass import readClass

def plotSamples(namePickle,nameFile,dim,mf):
    if mf == 1:
        mf = True
    else:
        mf = False
    dbn = readClass(namePickle)
    samples = dbn.generate_visible_samples(mean_field = mf)
    plotSamples_DBN(samples, nameFile, dim)

def plotSamples_DBN(obj, name, dim = 28, nrows = 10, ncols = 10):
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
                        help="file name of output png <default: 'test'>")
    parser.add_argument("--dim", type=int, default=28,
                        help="square dimensions on which to remap images <default: 28>")
    parser.add_argument("--mean-field", type=int, default=1,
                        help="draw actual samples (0) or mean-field samples (1) <default: 1>")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-p', '--pickle', type=str, 
                               help="name of directory where dbn.pickle is stored",
                               required=True)
    args = parser.parse_args()
    plotSamples(args.pickle,args.out,args.dim,args.mean_field)

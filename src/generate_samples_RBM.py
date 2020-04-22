#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import glob
from random import sample
from obj.RBM import RBM
from aux.updateClass import readClass
from skimage.transform import resize

def plotSamples(namePickle,nameFile,dim,mf,number_samples=100,indices = None):
    if mf == 1:
        mf = True
    else:
        mf = False
    rbm = readClass(namePickle)
    if "fashion_mnist" in namePickle:
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, _), (_,_) = fashion_mnist.load_data()
        x_train = x_train/255.0
        x_train = [tf.cast(tf.reshape(x,shape=(784,1)),"float32") for x in x_train]
    elif "mnist" in namePickle:
        mnist = tf.keras.datasets.mnist
        (x_train, _), (_,_) = mnist.load_data()
        x_train = x_train/255.0
        x_train = [tf.cast(tf.reshape(x,shape=(784,1)),"float32") for x in x_train]
    elif "faces" in namePickle:
        x_train = [resize(mpimg.imread(file),(28,28)) for file in glob.glob("data/faces/*")]
        x_train = np.asarray(x_train)
        # make images sparse for easier distinctions
        for img in x_train:
            img[img < np.mean(img)+0.5*np.std(img)] = 0
        x_train = [tf.cast(tf.reshape(x,shape=(784,1)),"float32") for x in x_train]
    else:
        raise NameError("unknown data type: %s" % x_train)
    # run gibbs sampling 15 times
    if indices is None:
        indices = sample(range(len(x_train)), number_samples)
    x_train = [x_train[ind] for ind in indices]
    samples = rbm.gibbs_sampling(x_train, k = 15)
    if mf == True:
       samples = rbm.prop_down(rbm.prop_up(samples))
    plotSamples_RBM(samples, nameFile, dim)

def plotSamples_RBM(obj, name, dim = 28, nrows = 10, ncols = 10):
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
                               help="name of directory where rbm.pickle is stored",
                               required=True)
    args = parser.parse_args()
    plotSamples(args.pickle,args.out,args.dim,args.mean_field)

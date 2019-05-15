#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import datetime
import argparse
import re
import glob
from obj.DBN import DBN
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize

################################
# train DBN from input data
################################

def trainDBN(data, learning_rate, k1, k2, epochs, batch_size, dims):
    # import data
    print("importing training data")
    if data == "fashion_mnist":
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (_,_) = fashion_mnist.load_data()
    elif data == "mnist":
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (_,_) = mnist.load_data()
    elif data == "faces":
        x_train = [resize(mpimg.imread(file),(28,28)) for file in glob.glob("data/faces/*")]
        x_train = np.asarray(x_train)
        # make images sparse for easier distinctions
        for img in x_train:
            img[img < np.mean(img)+0.5*np.std(img)] = 0
    else:
        raise NameError("unknown data type: %s" % data)
    if data == "mnist" or data == "fashion_mnist":
        x_train = x_train/255.0
        x_train = [tf.cast(tf.reshape(x,shape=(784,1)),"float32") for x in x_train]
    elif data == "faces":
        # auto conversion to probabilities in earlier step
        x_train = [tf.cast(tf.reshape(x,shape=(784,1)),"float32") for x in x_train]
    # create log directory
    current_time = getCurrentTime()+"_"+re.sub(",","_",dims)+"_"+data+"_dbn"
    os.makedirs("pickles/"+current_time)
    # parse string input into integer list
    dims = [int(el) for el in dims.split(",")]
    dbn = DBN(dims, learning_rate, k1, k2, epochs, batch_size)
    dbn.train_PCD(x_train)
    # dump dbn pickle
    f = open("pickles/"+current_time+"/dbn.pickle", "wb")
    pickle.dump(dbn, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

####################################
# main command call
####################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mnist",
                        help="data source to train DBN, possibilities are 'mnist', 'fashion_mnist' and 'faces', defaults to 'mnist'")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="learning rate for stacked RBMs, defaults to 0.01")
    parser.add_argument("--k1", type=int, default=1,
                        help="number of Gibbs-sampling steps pre-PCD-k algorithm, defaults to 1")
    parser.add_argument("--k2", type=int, default=5,
                        help="number of Gibbs-sampling steps during PCD-k algorithm, defaults to 5")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of overall training data passes for each RBM, defaults to 1")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="size of training data batches, defaults to 5")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-d', '--dimensions', type=str, 
                               help="consecutive enumeration of visible and hidden layers separated by a comma character, eg. 784,500,784,500", 
                               required=True)
    args = parser.parse_args()
    # train DBN based on parameters
    trainDBN(args.data,args.learning_rate,args.k1,args.k2,args.epochs,args.batch_size,args.dimensions)

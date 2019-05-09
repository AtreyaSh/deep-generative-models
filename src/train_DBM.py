#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import datetime
import argparse
import re
from obj.DBM import DBM
import tensorflow as tf

################################
# train DBM from MNIST
################################

def trainDBM(data, learning_rate, k1, k2, k3, epochs, batch_size, dims):
    # import mnist data
    print("importing MNIST training data")
    if data == "fashion_mnist":
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (_,_) = fashion_mnist.load_data()
    elif data == "mnist":
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (_,_) = mnist.load_data()
    else:
        raise NameError("unknown data type: %s" % data)
    x_train = x_train/255.0
    x_train = [tf.cast(tf.reshape(x,shape=(784,1)),"float32") for x in x_train]
    # create log directory
    current_time = getCurrentTime()+"_"+re.sub(",","_",dims)+"_"+data
    os.makedirs("pickles/"+current_time)
    # parse string input into integer list
    dims = [int(el) for el in dims.split(",")]
    dbm = DBM(dims, learning_rate, k1, k2, k3, epochs, batch_size)
    dbm.train_PCD(x_train)
    # dump dbm pickle
    f = open("pickles/"+current_time+"/dbm.pickle", "wb")
    pickle.dump(dbm, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

####################################
# main command call
####################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mnist",
                        help="data source to train DBM, possibilities are 'mnist' and 'fashion_mnist', defaults to 'mnist'")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="learning rate for stacked RBMs, defaults to 0.01")
    parser.add_argument("--k1", type=int, default=1,
                        help="number of Gibbs-sampling steps pre-PCD-k algorithm, defaults to 1")
    parser.add_argument("--k2", type=int, default=5,
                        help="number of Gibbs-sampling steps during PCD-k algorithm, defaults to 5")
    parser.add_argument("--k3", type=int, default=5,
                        help="number of Gibbs-sampling steps before transferring samples to next model, defaults to 5")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of overall training data passes for each RBM, defaults to 1")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="size of training data batches, defaults to 5")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-d', '--dimensions', type=str, 
                               help="consecutive enumeration of visible and hidden layers separated by a comma character, eg. 784,500,500,1000", 
                               required=True)
    args = parser.parse_args()
    # train DBM based on parameters
    trainDBM(args.data,args.learning_rate,args.k1,args.k2,args.k3,args.epochs,args.batch_size,args.dimensions)

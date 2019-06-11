#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import datetime
import argparse
import re
import csv
import tensorflow as tf
import numpy as np
import glob
from skimage.transform import resize
import matplotlib.image as mpimg
from obj.CGAN import CGAN

################################
# train CGAN from input data
################################

def trainCGAN(data,learning_rate,epochs,batch_size,im_dim,num_filters,latent_dimensions,
              g_factor,drop_rate):
    # import data
    print("importing training data")
    if data == "fashion_mnist":
        (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
    elif data == "mnist":
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    elif data == "faces":
        train_images = [resize(mpimg.imread(file),(28,28)) for file in glob.glob("./data/faces/*")]
        train_images = np.asarray(train_images,dtype="float32")
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    else:
        raise NameError("unknown data type: %s" % data)
    if data == "mnist" or data == "fashion_mnist":
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images /= 255.
    # create log directory
    current_time = getCurrentTime()+"_"+re.sub(",","_",str(latent_dimensions))+"_"+data+"_cgan"
    os.makedirs("pickles/"+current_time)
    # create model
    model = CGAN(latent_dim=latent_dimensions, epochs = epochs, batch_size = batch_size, 
                 learning_rate = learning_rate, im_dim = im_dim, n_filters = num_filters,
                 g_factor = g_factor, drop_rate = drop_rate)
    model.train(train_images)
    # save model
    model.save_weights("pickles/"+current_time+"/cgan")
    csvfile = open('pickles/'+ current_time + '/' + 'log.csv', 'w')
    fieldnames = ["data", "learning_rate", "epochs", "batch_size", "im_dim", "num_filters", 
                  "latent_dimensions", "g_factor", "drop_rate"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"data":data, "learning_rate":learning_rate, "epochs":epochs, 
                     "batch_size":batch_size, "im_dim":im_dim, "num_filters":num_filters, 
                     "latent_dimensions":latent_dimensions, "g_factor":g_factor,
                     "drop_rate":drop_rate})
    csvfile.close()

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

####################################
# main command call
####################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="mnist",
            help="data source to train CGAN, possibilities are 'mnist', 'fashion_mnist' and 'faces' <default: 'mnist'>")
    parser.add_argument("--learning-rate", type=float, default=0.001,
            help="learning rate, <default: 0.001>")
    parser.add_argument("--epochs", type=int, default=50,
            help="number of epochs for training <default: 50>")
    parser.add_argument("--batch-size", type=int, default=100,
            help="size of training data batches <default: 100>")
    parser.add_argument("--im-dim", type=int, default=28,
            help="square dimensionality of input images <default: 28>")
    parser.add_argument("--num-filters", type=int, default=32,
            help="number of filters to be used in convolutional layers <default: 32>")
    parser.add_argument("--g-factor", type=float, default=1,
            help=" scalar multiple by which learning rate for the generator is multiplied <default: 1>")
    parser.add_argument("--drop-rate", type=float, default=0.5,
            help="dropout rate for hidden dense layers <default: 0.5>")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-l', '--latent-dimensions', type=int,
                               help="number of central latent dimensions in CGAN, 2 dimensions are recommended for quick manifold visualization", 
                               required=True)
    args = parser.parse_args()
    # train CGAN based on parameters
    trainCGAN(args.data,args.learning_rate,args.epochs,args.batch_size,args.im_dim,
              args.num_filters,args.latent_dimensions,args.g_factor,args.drop_rate)

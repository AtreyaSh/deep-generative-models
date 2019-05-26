#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import imageio
import pandas as pd
from obj.VAE import CVAE

##############################
# plot VAE manifold
##############################

def squareToSpiral(d):
    D = d-1
    res = [0]
    while D >= 0:
        # downwards
        if D != d-1:
            res.extend([res[len(res)-1] + 1])
        res.extend([res[len(res)-1]+(i+1) for i in range(D)])
        # rightwards
        res.extend([res[len(res)-1]+(i+1)*d for i in range(D)])
        # upwards
        res.extend([res[len(res)-1]-(i+1) for i in range(D)])
        # leftwards    
        res.extend([res[len(res)-1]-(i+1)*d for i in range(D-1)])
        # update counter, makes move for even and odd respectively
        D -= 2
    return res

def plotManifold_VAE(namePickle,nameFile,im_dim,grid_size,latent_range,std,num_samples):
    print("plotting manifold samples")
    dim = pd.read_csv("./pickles/"+namePickle+"/log.csv")["latent_dimensions"][0]
    model = CVAE(dim)
    model.load_weights("./pickles/"+namePickle+"/vae")
    nx = ny = grid_size
    x_values = np.linspace(-latent_range, latent_range, nx)
    y_values = np.linspace(-latent_range, latent_range, ny)
    canvas = np.empty((im_dim*ny, im_dim*nx))
    s = [std for i in range(dim)]
    load=[]
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            mean = [xi for i in range(int(dim/2))]+[yi for i in range(int(np.ceil(dim/2)))]
            x_mean = tf.reduce_mean(model.sample(tf.random.normal((num_samples,model.latent_dim),mean=mean,stddev=s)),axis=0)[:,:,0]
            load.append(x_mean)
            canvas[(nx-i-1)*im_dim:(nx-i)*im_dim, j*im_dim:(j+1)*im_dim] = x_mean
    plt.figure(figsize=(8, 10))
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.savefig(os.path.abspath(os.path.dirname(os.getcwd()))+"/img/"+nameFile+".png", dpi=400)
    # create gif
    print("creating transition gif")
    indices = squareToSpiral(grid_size)
    load = [load[i].numpy() for i in indices]
    kargs = {'duration': 0.01}
    imageio.mimsave(os.path.abspath(os.path.dirname(os.getcwd()))+"/img/"+nameFile+".gif",load, **kargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test",
                        help="file name of output png, defaults to 'test'")
    parser.add_argument("--im-dim", type=int, default=28,
                        help="square dimensions on which to remap images, defaults to 28")
    parser.add_argument("--grid-size", type=int, default=40,
                        help="square dimensions or sensitivity of grid plot, defaults to 40")
    parser.add_argument("--latent-range", type=float, default=3,
                        help="range on which to search manifold mean, defaults to 3")
    parser.add_argument("--std", type=float, default=0.01,
                        help="standard deviation of latent distribution, defaults to 0.01")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="number of averaging samples per plot cell, defaults to 50")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-p', '--pickle', type=str,
                               help="name of directory where vae weights are stored",
                               required=True)
    args = parser.parse_args()
    plotManifold_VAE(args.pickle,args.out,args.im_dim,args.grid_size,args.latent_range,args.std,args.num_samples)

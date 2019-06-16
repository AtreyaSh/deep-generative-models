#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D 
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def plotStats(file):
    # read in text and remove trailing newline characters
    with open(file, "r") as text:
        store = [line.rstrip().split(",") for line in text]
    # rename file
    file = re.sub(r"\..*", "", file)
    # remove epoch delineations
    store = [line for line in store if len(line) != 1]
    # split store into 3 components
    step, disc, gen = [], [], []
    for el in store:
        for obj in el:
            if "Epoch" in obj:
                step.append(el)
                break
            elif "discriminator" in obj:
                disc.append(el)
                break
            elif "generator" in obj:
                gen.append(el)
                break
            else:
                pass    
    # flatten step into batch iterations
    step = [[re.sub(r'[^\d]*', '', el[0])] + re.sub(r'[^\d\/]*', '', el[1]).split("/") for el in step]
    step = np.array([((int(el[0])-1)*int(el[2]))+int(el[1]) for el in step])
    # parse discriminator into loss and gradient
    disc = np.array([[float(re.sub(r'[^\d\.]*', '', el[0]))] + 
                      [float(re.sub(r'[^\d\.]*', '', el[1]))] for el in disc])
    # parse discriminator into loss and gradient
    gen = np.array([[float(re.sub(r'[^\d\.]*', '', el[0]))] + 
                     [float(re.sub(r'[^\d\.]*', '', el[1]))] for el in gen])
    # 2d line plot for loss/gradient-time
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,6))
    axes = fig.add_subplot(111, frameon=False)
    for i, col in enumerate(ax):      
            col.grid(alpha = 0.5)
            col.set_axisbelow(True)
            col.plot(step,disc[:,i], color = "red", alpha = 0.6, label = "Discriminator")
            col.plot(step,gen[:,i], color = "green", alpha = 0.8, label = "Generator")
            if i == 0:
                col.set_ylabel(r"Log-Loss", labelpad = 10)
            elif i == 1:
                col.set_ylabel(r"Log-Loss Gradient", labelpad = 10)
            col.legend()
    axes.set_xlabel(r"Batch Iteration",labelpad = 10)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.savefig(file+"_2d.png", dpi = 200)
    # 3d scatterplot for loss-gradient-step
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(disc[:,0],disc[:,1],step, c=step,cmap="inferno",s=30)
    ax.set_title(r"\textbf{Discriminator}", pad = 20,  fontsize = 15)
    ax.set_xlabel(r'Log-Loss', labelpad = 10, fontsize = 12)
    ax.set_ylabel(r'Log-Loss Gradient',labelpad = 10, fontsize = 12)
    ax.set_zlabel(r'Batch Iteration',labelpad = 10, fontsize = 12)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlim([0,14])
    ax.set_title(r"\textbf{Generator}", pad = 20, fontsize = 15)
    ax.scatter(gen[:,0],gen[:,1],step, c=step,cmap="inferno",s=30)  
    ax.set_xlabel(r'Log-Loss', labelpad = 10, fontsize = 12)
    ax.set_ylabel(r'Log-Loss Gradient',labelpad = 10, fontsize = 12)
    ax.set_zlabel(r'Batch Iteration',labelpad = 10, fontsize = 12)
    plt.tight_layout()
    plt.savefig(file+"_3d.png", dpi = 200)
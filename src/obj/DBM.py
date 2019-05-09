#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from random import sample 
from tqdm import tqdm
from .RBM import RBM
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

class DBM:
    """
    Deep Boltzmann Machine (RBM) in TensorFlow 2
    pseudocode adapted from Hugo Larochelle's deep-learning Youtube series "Neural networks [7.9]"
    """
    def __init__(self, dims, learning_rate = 0.01, k1 = 1, k2 = 5, k3 = 5, epochs = 1, batch_size = 5):
        """ initialize stacked RBMs """
        self.models = [RBM(num_visible=dims[i],num_hidden=dims[i+1],
                           learning_rate=learning_rate, k1=k1, k2=k2, epochs=epochs,
                           batch_size=batch_size) for i in range(len(dims)-1)]
        self.k3 = k3
        self.top_samples = None

    def train_PCD(self, data):
        """ train stacked RBMs via greedy PCD-k algorithm """
        for i in range(len(self.models)):
            print("Training RBM: %s" % str(i+1))
            self.models[i].persistive_contrastive_divergence_k(data)
            if i != len(self.models)-1:
                print("Sampling data for model: %s" % str(i+2))
                self.models[i+1].b_v = self.models[i].b_h
                data = [self.models[i].prop_up(self.models[i].gibbs_sampling(img,self.k3)) for img in tqdm(data)]
            else:
                print("Final model, no generation for next model")
                self.top_samples = data

    def generate_visible_downwards(self, samples = None, number_samples = None, k = 15):
        """ generate visible samples from last RBM#s input samples """
        print("Gibbs sampling at deepest model: %s" % str(len(self.models)))
        if samples == None:
            samples = self.top_samples
        if number_samples == None:
            number_samples = len(samples)
        new_data = [self.models[len(self.models)-1].gibbs_sampling(img,k) for img in tqdm(sample(samples,number_samples))]
        for i in reversed(range(len(self.models)-1)):
            print("Downward propagation at model: %s" % str(i+1))
            new_data = [self.models[i].prop_down(img) for img in new_data]
        return new_data

    def generate_visible_up_down(self, test_data, number_samples = None, k = 15):
        """ propagate new samples upwards and create visible samples through downward propagation """
        for i in range(len(self.models)-1):
            print("Upward propagation at model: %s" % str(i+1))
            test_data = [self.models[i].prop_up(img) for img in test_data]
        return self.generate_visible_downwards(test_data, number_samples, k)
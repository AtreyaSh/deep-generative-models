#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
import tensorflow as tf
from random import sample 
from .RBM import RBM

class DBN:
    """
    Deep Belief Network (DBN) in TensorFlow 2
    pseudocode adapted from Hugo Larochelle's deep-learning Youtube series "Neural networks [7.9]"
    """
    def __init__(self, dims, learning_rate = 0.01, k1 = 1, k2 = 5, epochs = 1, batch_size = 5):
        """ initialize stacked RBMs """
        self.models = [RBM(num_visible=dims[i],num_hidden=dims[i+1],
                           learning_rate=learning_rate, k1=k1, k2=k2, epochs=epochs,
                           batch_size=batch_size) for i in range(len(dims)-1)]
        self.top_samples = None

    def train_PCD(self, data):
        """ train stacked RBMs via greedy PCD-k algorithm """
        for i in range(len(self.models)):
            print("Training RBM: %s" % str(i+1))
            self.models[i].persistive_contrastive_divergence_k(data)
            if i != len(self.models)-1:
                print("Sampling data for model: %s" % str(i+2))
                self.models[i+1].b_v = tf.Variable(self.models[i].b_h)
                if self.models[i+1].w.get_shape().as_list() == tf.transpose(self.models[i].w).get_shape().as_list():
                    print("Assigning previously learned transpose-weights to next model")
                    self.models[i+1].w = tf.Variable(tf.transpose(self.models[i].w))
                    self.models[i+1].b_h = tf.Variable(self.models[i].b_v)
                data = [self.models[i].random_sample(self.models[i].prop_up(img)) for img in data]
            else:
                print("Final model, no generation for next model")
                self.top_samples = data

    def generate_visible_samples(self, k = 15, number_samples = 100, indices = None, mean_field = True):
        """ generate visible samples from last RBMs input samples """
        print("Gibbs sampling at inner RBM: %s" % str(len(self.models)))
        samples = self.top_samples
        if indices is None:
            new_data = [self.models[len(self.models)-1].gibbs_sampling(img,k) for img in sample(samples,number_samples)]
        else:
            samples = [samples[i] for i in indices]
            new_data = [self.models[len(self.models)-1].gibbs_sampling(img,k) for img in samples]
        for i in reversed(range(len(self.models)-1)):
            print("Downward propagation at model: %s" % str(i+1))
            if mean_field:
                new_data = [self.models[i].prop_down(img) for img in new_data]
            else:
                new_data = [self.models[i].random_sample(self.models[i].prop_down(img)) for img in new_data]
        return new_data

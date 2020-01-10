#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
import tensorflow as tf
from random import sample
from .RBM import RBM

class DBM:
    """
    Deep Boltzmann Machine (DBM) in TensorFlow 2
    adapted from Ruslan Salakhutdinov's PhD thesis (University of Toronto) pp. 58
    """
    def __init__(self, dims, learning_rate = 0.01, k1 = 1, k2 = 5, epochs = 1, batch_size = 5):
        """ initialize stacked RBMs """
        self.models = [RBM(num_visible=dims[i],num_hidden=dims[i+1],
                           learning_rate=learning_rate, k1=k1, k2=k2, epochs=epochs,
                           batch_size=batch_size) for i in range(len(dims)-1)]
        self.data = []

    def train_PCD(self, data):
        """ train stacked RBMs via greedy PCD-k algorithm """
        # save training data into base layer
        self.data.append(data)
        for i in range(len(self.models)):
            print("Training RBM: %s" % str(i+1))
            if i == 0:
                print("Bottom RBM")
                self.models[i].persistive_contrastive_divergence_k(data, position = "bottom")
            elif i == len(self.models)-1:
                print("Top RBM")
                self.models[i].persistive_contrastive_divergence_k(data, position = "top")
            else:
                print("Intermediate RBM")
                self.models[i].w = tf.Variable(2*self.models[i].w)
                self.models[i].b_h = tf.Variable(2*self.models[i].b_h)
                self.models[i].b_v = tf.Variable(2*self.models[i].b_v)
                self.models[i].persistive_contrastive_divergence_k(data)
                self.models[i].w = tf.Variable(0.5*self.models[i].w)
                self.models[i].b_h = tf.Variable(0.5*self.models[i].b_h)
                self.models[i].b_v = tf.Variable(0.5*self.models[i].b_v)
            if i != len(self.models)-1:
                print("Sampling data for model: %s" % str(i+2))
                self.models[i+1].b_v = tf.Variable(self.models[i].b_h)
                if self.models[i+1].w.get_shape().as_list() == tf.transpose(self.models[i].w).get_shape().as_list():
                    print("Assigning previously learned transpose-weights to next model")
                    self.models[i+1].w = tf.Variable(tf.transpose(self.models[i].w))
                    self.models[i+1].b_h = tf.Variable(self.models[i].b_v)
            data = [self.models[i].random_sample(self.models[i].prop_up(img,q=2)) for img in data]
            self.data.append(data)

    def block_gibbs_sampling(self, k = 15, indices = None, number_samples = 100, mean_field = True):
        """ generate samples using even/odd layers which compromise overall Gibbs sampling """
        data = self.data[0]
        if indices is None:
            indices = sample(range(len(data)), number_samples)
        data = [data[ind] for ind in indices]
        for q in range(k):
            if q == k-1 and mean_field:
                # upward pass
                for i in range(len(self.models)):
                    if i != len(self.models)-1:
                        data = [tf.nn.sigmoid(tf.add(self.models[i].prop_up(data[j],sig=False),
                                       self.models[i+1].prop_down(self.data[i+2][indices[j]],sig=False))) for j in range(len(data))]
                    else:
                        data = [self.models[i].prop_up(data[j]) for j in range(len(data))]
                # downward pass
                for i in reversed(range(len(self.models))):
                    if i != 0:
                        data = [tf.nn.sigmoid(tf.add(self.models[i].prop_down(data[j],sig=False),
                                       self.models[i-1].prop_up(self.data[i-1][indices[j]],sig=False))) for j in range(len(data))]
                    else:
                        data = [self.models[i].prop_down(data[j]) for j in range(len(data))]
            else:
                # upward pass
                for i in range(len(self.models)):
                    if i != len(self.models)-1:
                        data = [self.models[i].random_sample(tf.nn.sigmoid(tf.add(self.models[i].prop_up(data[j],sig=False),
                                       self.models[i+1].prop_down(self.data[i+2][indices[j]],sig=False)))) for j in range(len(data))]
                    else:
                        data = [self.models[i].random_sample(self.models[i].prop_up(data[j])) for j in range(len(data))]
                # downward pass
                for i in reversed(range(len(self.models))):
                    if i != 0:
                        data = [self.models[i].random_sample(tf.nn.sigmoid(tf.add(self.models[i].prop_down(data[j],sig=False),
                                       self.models[i-1].prop_up(self.data[i-1][indices[j]],sig=False)))) for j in range(len(data))]
                    else:
                        data = [self.models[i].random_sample(self.models[i].prop_down(data[j])) for j in range(len(data))]
        return data

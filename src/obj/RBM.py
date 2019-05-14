#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

class RBM:
    """
    Restricted Boltzmann Machine (RBM) in TensorFlow 2
    pseudocode adapted from Hugo Larochelle's2 deep-learning Youtube series "Neural networks [5.6]"
    """
    def __init__(self, num_visible, num_hidden, learning_rate = 0.01, k1 = 1, k2 = 5, epochs = 1, batch_size = 5):
        """ initialize weights/biases randomly """
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        # initialize starting points for weights and biases
        self.w = tf.Variable(tf.random.uniform(shape=(num_hidden,num_visible), maxval=1, minval=-1),dtype="float32")
        # bias over hidden variables
        self.b_h = tf.Variable(tf.random.uniform(shape=(num_hidden,1), maxval=1, minval=-1),dtype="float32")
        # bias over visible variables
        self.b_v = tf.Variable(tf.random.uniform(shape=(num_visible,1), maxval=1, minval=-1),dtype="float32")
        # set training parameters
        self.learning_rate = learning_rate
        self.k1 = k1
        self.k2 = k2
        self.epochs = epochs
        self.batch_size = batch_size

    # @tf.function
    def gibbs_sampling(self, v, k = 5):
        """ function for gibbs sampling for k-iterations """
        for _ in range(k):
            binomial_probs = self.prop_up(v)
            h = self.random_sample(binomial_probs)
            binomial_probs = self.prop_down(h)
            v = self.random_sample(binomial_probs)
        return v
    
    def random_sample(self, input_v):
        """ generate binary samples given probability vector """
        return tf.where(tf.random.uniform(shape=tf.shape(input_v)) - input_v < 0,
                         tf.ones(tf.shape(input_v)), tf.zeros(tf.shape(input_v)))
        
    def prop_up(self, v):
        """ upwards mean-field propagation """
        return tf.nn.sigmoid(tf.add(self.b_h, tf.matmul(self.w,v)))

    def prop_down(self, h):
        """ downwards mean-field propagation """
        return tf.sigmoid(tf.add(self.b_v, tf.matmul(tf.transpose(self.w),h)))

    def chunks(self, l, n):
        """ create chunks/batches from input data """
        for i in range(0, len(l), n):
            yield l[i:i + n]
    
    # @tf.function
    def contrastive_divergence_k(self, tensors):
        """ learn and update weights/biases via PCD-k algorithm """
        tensors = list(self.chunks(tensors,self.batch_size))
        num_samples = len(tensors)
        for i in range(self.epochs):
            j = 0
            log = 0
            print("Epoch: %s" % str(i+1))
            for batch in tensors:
                if j % 20 == 0:
                    print("Batch number: %s/%d" % (j,num_samples))
                    if j != 0:
                        tf.print("Mean weight-gradient 2-norm: ", log/j)
                # compute starting gradient
                batch = tf.stack(batch)
                u = tf.map_fn(self.prop_up,batch)
                g = tf.reduce_mean(tf.stack([tf.matmul(u[i],tf.transpose(batch[i])) for i in range(self.batch_size)]),0)
                # compute sampled gibbs
                v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k2),batch)
                u_new = tf.map_fn(self.prop_up,v_new)
                # compute change to gradient, average gradient shifts before adding
                g_delta = -1*tf.reduce_mean(tf.stack([tf.matmul(u_new[i],tf.transpose(v_new[i])) for i in range(self.batch_size)]),0)
                g += g_delta
                # log norm of new gradient
                log += tf.norm(g,ord=2)
                j += 1
                # update parameters
                self.w.assign_add(self.learning_rate*g)
                self.b_h.assign_add(self.learning_rate*tf.reduce_mean(tf.add(u,-1*u_new),0))
                self.b_v.assign_add(self.learning_rate*tf.reduce_mean(tf.add(batch,-1*v_new),0))
            tf.print("Mean weight-gradient 2-norm:", log/num_samples)
    
    # @tf.function
    def persistive_contrastive_divergence_k(self, tensors):
        """ learn and update weights/biases via PCD-k algorithm """
        tensors = list(self.chunks(tensors,self.batch_size))
        num_samples = len(tensors)
        for i in range(self.epochs):
            j = 0
            log = 0
            print("Epoch: %s" % str(i+1))
            for batch in tensors:
                if j % 20 == 0:
                    print("Batch number: %s/%d" % (j,num_samples))
                    if j != 0:
                        tf.print("Mean weight-gradient 2-norm: ", log/j)
                # compute starting gradient
                batch = tf.stack(batch)
                u = tf.map_fn(self.prop_up,batch)
                g = tf.reduce_mean(tf.stack([tf.matmul(u[i],tf.transpose(batch[i])) for i in range(self.batch_size)]),0)
                # compute sampled gibbs
                batch = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k1),batch)
                v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k2),batch)
                u_new = tf.map_fn(self.prop_up,v_new)
                # compute change to gradient, average gradient shifts before adding
                g_delta = -1*tf.reduce_mean(tf.stack([tf.matmul(u_new[i],tf.transpose(v_new[i])) for i in range(self.batch_size)]),0)
                g += g_delta
                # log norm of new gradient
                log += tf.norm(g,ord=2)
                j += 1
                # update parameters
                self.w.assign_add(self.learning_rate*g)
                self.b_h.assign_add(self.learning_rate*tf.reduce_mean(tf.add(u,-1*u_new),0))
                self.b_v.assign_add(self.learning_rate*tf.reduce_mean(tf.add(batch,-1*v_new),0))
            tf.print("Mean weight-gradient 2-norm:", log/num_samples)
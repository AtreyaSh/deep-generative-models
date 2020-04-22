#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
import tensorflow as tf

class RBM:
    """
    Restricted Boltzmann Machine (RBM) in TensorFlow 2
    pseudocode adapted from Hugo Larochelle's deep-learning Youtube series "Neural networks [5.6]"
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

    def gibbs_sampling(self, v, k = 5, q1=1, q2=1):
        """ function for gibbs sampling for k-iterations """
        for _ in range(k):
            binomial_probs = self.prop_up(v,q1)
            h = self.random_sample(binomial_probs)
            binomial_probs = self.prop_down(h,q2)
            v = self.random_sample(binomial_probs)
        return v

    def random_sample(self, input_v):
        """ generate binary samples given probability vector """
        return tf.where(tf.random.uniform(shape=tf.shape(input_v)) - input_v < 0,
                         tf.ones(tf.shape(input_v)), tf.zeros(tf.shape(input_v)))

    def prop_up(self, v, q=1, sig = True):
        """ upwards mean-field propagation """
        if sig:
            return tf.nn.sigmoid(q*tf.add(self.b_h, tf.matmul(self.w,v)))
        else:
            return tf.add(self.b_h, tf.matmul(self.w,v))

    def prop_down(self, h, q=1, sig = True):
        """ downwards mean-field propagation """
        if sig:
            return tf.nn.sigmoid(q*tf.add(self.b_v, tf.matmul(tf.transpose(self.w),h)))
        else:
            return tf.add(self.b_v, tf.matmul(tf.transpose(self.w),h))

    def chunks(self, l, n):
        """ create chunks/batches from input data """
        return [l[i:i + n] for i in range(0, len(l), n)]

    def adam(self,g,t,m=None,r=None):
        """
        adam gradient descent optimization
        adapted from https://wiseodd.github.io/techblog/2016/06/22/nn-optimization/
        """
        beta1 = .9
        beta2 = .999
        eps = 1e-8
        if m is None and r is None:
            m = tf.zeros(shape=tf.shape(g),dtype="float32")
            r = tf.zeros(shape=tf.shape(g),dtype="float32")
        m = beta1*m + (1.-beta1)*g
        r = beta2*r + (1.-beta2)*tf.math.square(g)
        m_k_hat = m/(1.-beta1**(t+1))
        r_k_hat = r/(1.-beta2**(t+1))
        out = tf.math.divide(m_k_hat,tf.math.add(tf.math.sqrt(r_k_hat),eps))
        return out, m, r

    # @tf.function
    def contrastive_divergence_k(self, tensors, position = None):
        """ learn and update weights/biases via CD-k algorithm """
        tensors = self.chunks(tensors,self.batch_size)
        num_samples = len(tensors)
        # augment last batch in case missing some data
        if len(tensors[num_samples-1]) != self.batch_size:
            diff = self.batch_size - len(tensors[num_samples-1])
            tensors[num_samples-1].extend(tensors[0][:diff])
        for i in range(self.epochs):
            j = 0
            log_g = 0
            log_h = 0
            log_v = 0
            print("Epoch: %s" % str(i+1))
            for batch in tensors:
                if j % 20 == 0:
                    print("Batch number: %s/%d" % (j,num_samples))
                    if j != 0:
                        tf.print("Mean gradient 2-norm:", tf.reduce_mean([log_g/j,log_h/j,log_v/j]))
                # compute starting gradient
                batch = tf.stack(batch)
                if position is None:
                    u = tf.map_fn(self.prop_up,batch)
                    g = tf.reduce_mean(tf.stack([tf.matmul(u[i],tf.transpose(batch[i])) for i in range(self.batch_size)]),0)
                    # compute sampled gibbs
                    v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k2),batch)
                    u_new = tf.map_fn(self.prop_up,v_new)
                    # compute change to gradient, average gradient shifts before adding
                    g_delta = -1*tf.reduce_mean(tf.stack([tf.matmul(u_new[i],tf.transpose(v_new[i])) for i in range(self.batch_size)]),0)
                elif position == "bottom":
                    u = tf.map_fn(lambda x: self.prop_up(x,q=2),batch)
                    g = tf.reduce_mean(tf.stack([tf.matmul(u[i],tf.transpose(batch[i])) for i in range(self.batch_size)]),0)
                    # compute sampled gibbs
                    v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k2,q1=2,q2=1),batch)
                    u_new = tf.map_fn(lambda x: self.prop_up(x,q=2),v_new)
                    # compute change to gradient, average gradient shifts before adding
                    g_delta = -1*tf.reduce_mean(tf.stack([tf.matmul(u_new[i],tf.transpose(v_new[i])) for i in range(self.batch_size)]),0)
                elif position == "top":
                    u = tf.map_fn(lambda x: self.prop_up(x,q=1),batch)
                    g = tf.reduce_mean(tf.stack([tf.matmul(u[i],tf.transpose(batch[i])) for i in range(self.batch_size)]),0)
                    # compute sampled gibbs
                    v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k2,q1=1,q2=2),batch)
                    u_new = tf.map_fn(lambda x: self.prop_up(x,q=1),v_new)
                    # compute change to gradient, average gradient shifts before adding
                    g_delta = -1*tf.reduce_mean(tf.stack([tf.matmul(u_new[i],tf.transpose(v_new[i])) for i in range(self.batch_size)]),0)
                # update gradient and log result
                g += g_delta
                g_h = tf.reduce_mean(tf.add(u,-1*u_new),0)
                g_v = tf.reduce_mean(tf.add(batch,-1*v_new),0)
                # update gradient-logs
                log_g += tf.norm(g,ord=2)
                log_h += tf.norm(g_h,ord=2)
                log_v += tf.norm(g_v,ord=2)
                # perform adam operation to all gradients
                if j == 0:
                    g,m,r = self.adam(g,j)
                    g_h,m_h,r_h = self.adam(g_h,j)
                    g_v,m_v,r_v = self.adam(g_v,j)
                else:
                    g,m,r = self.adam(g,j,m,r)
                    g_h,m_h,r_h = self.adam(g_h,j,m_h,r_h)
                    g_v,m_v,r_v = self.adam(g_v,j,m_v,r_v)
                # update counter
                j += 1
                # update parameters
                self.w.assign_add(self.learning_rate*g)
                self.b_h.assign_add(self.learning_rate*g_h)
                self.b_v.assign_add(self.learning_rate*g_v)
            tf.print("Mean gradient 2-norm:", tf.reduce_mean([log_g/j,log_h/j,log_v/j]))

    # @tf.function
    def persistive_contrastive_divergence_k(self, tensors, position = None):
        """ learn and update weights/biases via PCD-k algorithm """
        tensors = self.chunks(tensors,self.batch_size)
        num_samples = len(tensors)
        # augment last batch in case missing some data
        if len(tensors[num_samples-1]) != self.batch_size:
            diff = self.batch_size - len(tensors[num_samples-1])
            tensors[num_samples-1].extend(tensors[0][:diff])
        for i in range(self.epochs):
            j = 0
            log_g = 0
            log_h = 0
            log_v = 0
            print("Epoch: %s" % str(i+1))
            for batch in tensors:
                if j % 20 == 0:
                    print("Batch number: %s/%d" % (j,num_samples))
                    if j != 0:
                        tf.print("Mean gradient 2-norm: ", tf.reduce_mean([log_g/j,log_h/j,log_v/j]))
                # compute starting gradient
                batch = tf.stack(batch)
                if position is None:
                    u = tf.map_fn(self.prop_up,batch)
                    g = tf.reduce_mean(tf.stack([tf.matmul(u[i],tf.transpose(batch[i])) for i in range(self.batch_size)]),0)
                    # compute sampled gibbs
                    if j == 0:
                        v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k1),batch)
                    v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k2),v_new)
                    u_new = tf.map_fn(self.prop_up,v_new)
                    # compute change to gradient, average gradient shifts before adding
                    g_delta = -1*tf.reduce_mean(tf.stack([tf.matmul(u_new[i],tf.transpose(v_new[i])) for i in range(self.batch_size)]),0)
                elif position == "bottom":
                    u = tf.map_fn(lambda x: self.prop_up(x,q=2),batch)
                    g = tf.reduce_mean(tf.stack([tf.matmul(u[i],tf.transpose(batch[i])) for i in range(self.batch_size)]),0)
                    # compute sampled gibbs
                    if j == 0:
                        v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k1,q1=2,q2=1),batch)
                    v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k2,q1=2,q2=1),v_new)
                    u_new = tf.map_fn(lambda x: self.prop_up(x,q=2),v_new)
                    # compute change to gradient, average gradient shifts before adding
                    g_delta = -1*tf.reduce_mean(tf.stack([tf.matmul(u_new[i],tf.transpose(v_new[i])) for i in range(self.batch_size)]),0)
                elif position == "top":
                    u = tf.map_fn(lambda x: self.prop_up(x,q=1),batch)
                    g = tf.reduce_mean(tf.stack([tf.matmul(u[i],tf.transpose(batch[i])) for i in range(self.batch_size)]),0)
                    # compute sampled gibbs
                    if j == 0:
                        v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k1,q1=1,q2=2),batch)
                    v_new = tf.map_fn(lambda x: self.gibbs_sampling(x,self.k2,q1=1,q2=2),v_new)
                    u_new = tf.map_fn(lambda x: self.prop_up(x,q=1),v_new)
                    # compute change to gradient, average gradient shifts before adding
                    g_delta = -1*tf.reduce_mean(tf.stack([tf.matmul(u_new[i],tf.transpose(v_new[i])) for i in range(self.batch_size)]),0)
                # update gradient and log result
                g += g_delta
                g_h = tf.reduce_mean(tf.add(u,-1*u_new),0)
                g_v = tf.reduce_mean(tf.add(batch,-1*v_new),0)
                # update gradient-logs
                log_g += tf.norm(g,ord=2)
                log_h += tf.norm(g_h,ord=2)
                log_v += tf.norm(g_v,ord=2)
                # perform adam operation to all gradients
                if j == 0:
                    g,m,r = self.adam(g,j)
                    g_h,m_h,r_h = self.adam(g_h,j)
                    g_v,m_v,r_v = self.adam(g_v,j)
                else:
                    g,m,r = self.adam(g,j,m,r)
                    g_h,m_h,r_h = self.adam(g_h,j,m_h,r_h)
                    g_v,m_v,r_v = self.adam(g_v,j,m_v,r_v)
                # update counter
                j += 1
                # update parameters
                self.w.assign_add(self.learning_rate*g)
                self.b_h.assign_add(self.learning_rate*g_h)
                self.b_v.assign_add(self.learning_rate*g_v)
            tf.print("Mean gradient 2-norm:", tf.reduce_mean([log_g/j,log_h/j,log_v/j]))

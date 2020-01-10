#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

class CVAE(tf.keras.Model):
    """
    Convolutional Variational Autoencoder (VAE)
    sub-class of tf.keras.Model
    code modified from TF2 CVAE tutorial: 
    https://www.tensorflow.org/alpha/tutorials/generative/cvae
    """
    def __init__(self, latent_dim, epochs = 5, batch_size = 50, learning_rate = 0.001,
                 im_dim = 28, n_filters = 32):
        """ initialize model layers and parameters """
        super(CVAE, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.latent_dim = latent_dim
        self.im_dim = im_dim
        self.n_filters = n_filters
        self.inference_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(int(self.im_dim), int(self.im_dim), 1)),
          tf.keras.layers.Conv2D(
              filters=int(self.n_filters), kernel_size=3, strides=(2, 2), activation='relu'),
          tf.keras.layers.Flatten(),
          # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
        ])
        self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units=(int((self.im_dim/2)**2)*self.n_filters), activation=tf.nn.relu),
          tf.keras.layers.Reshape(target_shape=(int(self.im_dim/2), int((self.im_dim/2)), self.n_filters)),
          tf.keras.layers.Conv2DTranspose(
              filters=int(self.n_filters),
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation='relu'),
          # No activation
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ])

    # @tf.function
    def encode(self, x):
        """ encode input data into log-normal distribution at latent layer """
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """ reparameterize normal distribution from learned mean/variance """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    # @tf.function
    def decode(self, z, apply_sigmoid=False):
        """ decode latent variables into visible samples """
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        else:
            return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        """ function for defining log normal PDF """
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                             axis=raxis)

    def compute_loss(self, x):
        """ compute ELBO loss given hyperparamters """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def compute_gradients(self, x):
        """ compute gradient given ELBO loss """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
            return tape.gradient(loss, self.trainable_variables), loss

    def apply_gradients(self, gradients):
        """ apply adam gradient descent optimizer for learning process """
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def train(self, train_dataset):
        """ main training call for CVAE """
        num_samples = int(train_dataset.shape[0]/self.batch_size)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(train_dataset.shape[0]).batch(self.batch_size)
        for i in range(self.epochs):
            j = 1
            norm = 0
            Loss = 0
            print("Epoch: %s" % str(i+1))
            for train_x in train_dataset:
                gradients, loss = self.compute_gradients(train_x)
                Loss += loss
                norm += tf.reduce_mean([tf.norm(g) for g in gradients])
                self.apply_gradients(gradients)
                if j != 1 and j % 20 == 0:
                    # good to print out euclidean norm of gradients
                    tf.print("Epoch: %s, Batch: %s/%s" % (i+1,j,num_samples))
                    tf.print("Mean-Loss: ", Loss/j, ", Mean gradient-norm: ", norm/j)
                j += 1

    def sample(self, eps=None, num = 50):
        """ sample latent layer and decode to generated visible """
        if eps is None:
            eps = tf.random.normal(shape=(num, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

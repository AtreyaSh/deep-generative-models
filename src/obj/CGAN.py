#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

class CGAN(tf.keras.Model):
    def __init__(self, latent_dim, epochs = 50, batch_size = 100, learning_rate = 0.001,
                 im_dim = 28, n_filters = 32, g_factor = 1, drop_rate = 0.5):
        """ initialize model layers and parameters """
        super(CGAN, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.g_factor = g_factor
        self.optimizer_d = tf.keras.optimizers.Adam(self.learning_rate)
        self.optimizer_g = tf.keras.optimizers.Adam(self.learning_rate*self.g_factor)
        self.latent_dim = latent_dim
        self.im_dim = im_dim
        self.n_filters = n_filters
        self.drop_rate = drop_rate
        self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dropout(self.drop_rate),
          tf.keras.layers.Dense(units=(int((self.im_dim/2)**2)*self.n_filters), activation="relu", name = "g1"),
          tf.keras.layers.Reshape(target_shape=(int(self.im_dim/2), int((self.im_dim/2)), self.n_filters)),
          tf.keras.layers.Conv2DTranspose(
              filters=int(self.n_filters),
              kernel_size=3,
              strides=(2, 2),
              padding="SAME",
              activation = "relu",
              name = "g2"),
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation = "relu", name = "g3"),
          tf.keras.layers.Conv2DTranspose(
              filters=1, kernel_size=1, strides=(1, 1), padding="SAME", activation="sigmoid", name = "g4"),
        ])
        self.discriminator_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(int(self.im_dim), int(self.im_dim), 1)),
          tf.keras.layers.Conv2D(
               filters=int(self.n_filters), kernel_size=3, strides=(2, 2), activation = tf.nn.leaky_relu, name = "d1"),
          tf.keras.layers.Conv2D(
               filters=int(self.n_filters), kernel_size=3, strides=(2, 2), activation = tf.nn.leaky_relu, name = "d2"),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dropout(self.drop_rate),
          tf.keras.layers.Dense(units = int((self.im_dim/4)**2), activation = tf.nn.leaky_relu, name = "d3"),
          tf.keras.layers.Dropout(self.drop_rate),
          tf.keras.layers.Dense(units = 1, name = "d4"),
        ])

    # @tf.function
    def generate(self, eps=None, num = None):
        """ generate fake sample using generative net """
        if num is None:
            num = self.batch_size
        if eps is None:
            eps = tf.random.normal(shape=(num, self.latent_dim))
        return self.generative_net(eps)

    # @tf.function
    def discriminate(self, x, apply_sigmoid=False):
        """ discriminate between fake and real samples """
        logits = self.discriminator_net(x)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        else:
            return logits

    def compute_loss_discriminator(self, x):
        """ compute log-loss for discriminator """
        x_logit = self.discriminate(x)
        gen = self.generate()
        gen_logit = self.discriminate(gen)
        cross_ent_x = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=tf.ones(shape=x_logit.shape))
        cross_ent_gen = tf.nn.sigmoid_cross_entropy_with_logits(logits=-gen_logit, labels=tf.ones(shape=gen_logit.shape))
        return tf.reduce_mean(cross_ent_gen) + tf.reduce_mean(cross_ent_x)

    def compute_loss_generator(self):
        """ compute log-loss for generator """
        gen = self.generate()
        gen_logit = self.discriminate(gen)
        cross_ent_gen = tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_logit, labels=tf.ones(shape=gen_logit.shape))
        return tf.reduce_mean(cross_ent_gen)

    def compute_gradients(self, x, sub):
        """
        compute dynamic gradients with separate optimizers which could theoretically undergo dynamic adjustment during training
        """
        with tf.GradientTape() as tape:
            if sub == "discriminator":
                loss = self.compute_loss_discriminator(x)
                t_v = [el for el in self.trainable_variables if "d" in el.name]
                return tape.gradient(loss, t_v), t_v, loss
            elif sub == "generator":
                loss = self.compute_loss_generator()
                t_v = [el for el in self.trainable_variables if "g" in el.name]
                return tape.gradient(loss, t_v), t_v, loss

    def apply_gradients(self, gradients, t_v, sub):
        """ apply adam gradient descent optimizer for learning process """
        if sub == "discriminator":
          self.optimizer_d.apply_gradients(zip(gradients, t_v))
        elif sub == "generator":
          self.optimizer_g.apply_gradients(zip(gradients, t_v))

    def train(self, train_dataset):
        """ main training call for CGAN """
        num_samples = int(train_dataset.shape[0]/self.batch_size)
        train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset).shuffle(train_dataset.shape[0]).batch(self.batch_size)
        for i in range(self.epochs):
            j = 1
            norm_d = 0
            Loss_d = 0
            norm_g = 0
            Loss_g = 0
            print("Epoch: %s" % str(i+1))
            for train_x in train_dataset:
                # disciminator step
                gradients, t_v, loss = self.compute_gradients(train_x, sub = "discriminator")
                Loss_d += loss
                norm_d += tf.reduce_mean([tf.norm(g) for g in gradients])
                self.apply_gradients(gradients, t_v, sub = "discriminator")
                # generator step
                gradients, t_v, loss = self.compute_gradients(train_x, sub = "generator")
                Loss_g += loss
                norm_g += tf.reduce_mean([tf.norm(g) for g in gradients])
                self.apply_gradients(gradients, t_v, sub = "generator")
                if j != 1 and j % 20 == 0:
                    # good to print out euclidean norm of gradients
                    tf.print("Epoch: %s, Batch: %s/%s" % (i+1,j,num_samples))
                    tf.print("Mean discriminator loss: ", Loss_d/j, ", Mean discriminator gradient norm: ", norm_d/j)
                    tf.print("Mean generator loss: ", Loss_g/j, ", Mean generator gradient norm: ", norm_g/j)
                j += 1

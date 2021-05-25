#!/usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.dense = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.dense(x)


def main(_):
    model = Model()

    inputs = np.random.uniform(0, 1, (1, 8))
    outputs = model(inputs)

    print(inputs)
    print(outputs)


if __name__ == "__main__":
    tf.compat.v1.app.run(main=main)

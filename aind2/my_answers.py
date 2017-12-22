#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
...


@author: ucaiado

Created on 12/22/2017
"""
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import keras


'''
Begin help functions
'''


'''
End help functions
'''


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
    wsize = window_size

    # slice the data
    X = [list(series[idx: idx+wsize]) for idx in range(len(series)-wsize)]
    y = series[wsize:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


def build_part1_RNN(window_size):
    '''
    Return a RNN to perform regression on our time series input/output data
    source: https://keras.io/getting-started/sequential-model-guide/

    :param window_size: integer. Bla
    '''
    model = Sequential()
    # layer 1 uses an LSTM module with 5 hidden units
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # layer 2 uses a fully connected module with one unit
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    return model


# TODO: return the text input with only ascii lowercase and the punctuation
# given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    return text


# TODO: fill out the function below that transforms the input text and
# window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs, outputs


# TODO build the required RNN model:  a single LSTM hidden layer with softmax
# activation, categorical_crossentropyloss
def build_part2_RNN(window_size, num_chars):
    pass

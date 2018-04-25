#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
...


@author: ucaiado

Created on 12/22/2017
"""
import numpy as np

<<<<<<< HEAD
from unidecode import unidecode
=======
# NOTE: Here I try to install unidecode
# to udacity-pa but it didnt work
try:
    from unidecode import unidecode
except (ImportError, NameError):
    import pip
    pip.main(['install', 'unidecode'])
# ##################################
>>>>>>> 00fd9a6e94eac185b00839d22c9e738d91feafa0
import string

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
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
    '''
    Return the input/output pairs using a slinding windows

    :param series: numpy array. a array of numbers
    :param window_size: integer. sliding window
    '''
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

    :param window_size: integer. sliding window
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
def cleaned_text_old(text):
    '''
    Return the text passed without non-ASCII characters.
    Keep same punctuation. When it is possible, remove accents
    from letters

    :param text: string. text to be used
    '''
    punctuation = [' ', '!', ',', '.', ':', ';', '?']
    text = text.lower()
    text = unidecode(text)
    # source: https://goo.gl/UB1vTa
    set_to_keep = set(string.ascii_lowercase + ''.join(punctuation))
    text = ''.join(filter(lambda s: s in set_to_keep, text))
    return text


# TODO: return the text input with only ascii lowercase and the punctuation
# given below included.
def cleaned_text(text):
    '''
<<<<<<< HEAD
    Return the text passed without non-ASCII characters.
    Keep same punctuation. When it is possible, remove accents
    from letters
=======
    Return the text passed without non-ASCII characters. Keep same punctuation.
>>>>>>> 00fd9a6e94eac185b00839d22c9e738d91feafa0

    :param text: string. text to be used
    '''
    punctuation = [' ', '!', ',', '.', ':', ';', '?']
    text = text.lower()
<<<<<<< HEAD
    text = unidecode(text)
=======
    # text = unidecode(text)
>>>>>>> 00fd9a6e94eac185b00839d22c9e738d91feafa0
    # source: https://goo.gl/UB1vTa
    set_to_keep = set(string.ascii_lowercase + ''.join(punctuation))
    text = ''.join(filter(lambda s: s in set_to_keep, text))
    return text


# TODO: fill out the function below that transforms the input text and
# window-size into a set of input/output pairs for use with our RNN model
def window_transform_text_old(text, window_size, step_size):
    '''
    Return the input/output pairs using a slinding windows with the passed
    step size

    :param text: string. text to be splited in input/output pairs
    :param window_size: integer. sliding window
    :param step_size: integer. charcter to jump at each iteration
    '''
    # containers for input/output pairs
    inputs = []
    outputs = []
    for x in range(window_size, len(text), step_size):
        inputs.append([y for y in text[:x][::-1][:window_size][::-1]])
        outputs.append(text[x])

    return inputs, outputs


def window_transform_text(text, window_size, step_size):
    '''
    Return the input/output pairs using a slinding windows with the passed
    step size

    :param text: string. text to be splited in input/output pairs
    :param window_size: integer. sliding window
    :param step_size: integer. charcter to jump at each iteration
    '''
    # containers for input/output pairs
    inputs = []
    outputs = []
<<<<<<< HEAD
    for x in range(window_size, len(text), step_size):
        inputs.append([y for y in text[:x][::-1][:window_size][::-1]])
        outputs.append(text[x])
=======
    for x in range(0, len(text)-window_size, step_size):
        inputs.append(text[x:(x + window_size)])
        outputs.append(text[x+window_size])
>>>>>>> 00fd9a6e94eac185b00839d22c9e738d91feafa0

    return inputs, outputs


# TODO build the required RNN model:  a single LSTM hidden layer with softmax
# activation, categorical_crossentropyloss
def build_part2_RNN(window_size, num_chars):
    '''
<<<<<<< HEAD
    Return a RNN to predict the next character after following any chunk of       
=======
    Return a RNN to predict the next character after following any chunk of
>>>>>>> 00fd9a6e94eac185b00839d22c9e738d91feafa0
    characters

    :param window_size: integer. sliding window
    :param num_chars: integer. number of unique characters to be considered
    '''
    model = Sequential()
    # layer 1 uses an LSTM module with 200 hidden units
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # layer 2 uses a linear module, fully connected
    model.add(Dense(num_chars, activation='linear'))
    # layer 3 uses a softmax activation
    model.add(Activation('softmax'))

    return model

#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
A starting template for writing unit tests to run and debug "unit" of codes,
as classes and functions. A folder of tests should be placed at each subfolder
of the source code repository. The test should be run from the top-level
directory of this folder calling "python -m tests.foo_test"

Source: Beginning test-driven development...(https://goo.gl/LwQNc4)

@author: ucaiado

Created on MM/DD/YYYY
"""

from unittest import TestCase

from aind.asl_data import AslDb
from aind.asl_utils import train_all_words
from aind.my_model_selectors import SelectorConstant
from aind.my_recognizer import recognize

FEATURES = ['right-y', 'right-x']


class TestRecognize(TestCase):
    def setUp(self):
        self.asl = AslDb()
        self.training_set = self.asl.build_training(FEATURES)
        self.test_set = self.asl.build_test(FEATURES)
        self.models = train_all_words(self.training_set, SelectorConstant)

    def test_recognize_probabilities_interface(self):
        probs, _ = recognize(self.models, self.test_set)
        self.assertEqual(len(probs), self.test_set.num_items, "Number of test items in probabilities list incorrect.")
        self.assertIn('FRANK', probs[0], "Dictionary of probabilities does not contain correct keys")
        self.assertIn('CHICKEN', probs[-1], "Dictionary of probabilities does not contain correct keys")

    def test_recognize_guesses_interface(self):
        _, guesses = recognize(self.models, self.test_set)
        self.assertEqual(len(guesses), self.test_set.num_items, "Number of test items in guesses list incorrect.")
        self.assertIsInstance(guesses[0], str, "The guesses are not strings")
        self.assertIsInstance(guesses[-1], str, "The guesses are not strings")


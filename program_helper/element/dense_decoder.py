# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


class DenseDecoder(object):
    def __init__(self, num_layers, units, initial_state, vocab_size):

        encoding = tf.layers.dense(initial_state, units, activation=tf.nn.tanh)
        for i in range(num_layers - 1):
            encoding = tf.layers.dense(encoding, units, activation=tf.nn.tanh)

        w = tf.get_variable('w', [units, vocab_size])
        b = tf.get_variable('b', [vocab_size])
        self.logits = tf.nn.xw_plus_b(encoding, w, b)


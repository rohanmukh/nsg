# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unlconfig.vocab.fp_dict_size,ess required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf


class DenseEncoder(object):
    def __init__(self, inputs, units, num_layers, output_units,
                 vocab_size, batch_size,
                 emb=None,
                 drop_prob=None):

        if drop_prob is None:
            drop_prob = tf.constant(0.0, dtype=tf.float32)

        '''
        inputs is of shape batch_size
        lets drop it randomly
        '''

        dropper = tf.cast(tf.random_uniform((batch_size, 1), 0, 1, dtype=tf.float32)
                          > drop_prob, tf.int32)
        inputs = inputs * tf.squeeze(dropper)

        '''
        now we will encode it
        '''
        emb = tf.get_variable('emb_ret', [vocab_size, units])
        emb_inp = tf.nn.embedding_lookup(emb, inputs)

        encoding = tf.layers.dense(emb_inp, units, activation=tf.nn.tanh)
        # encoding = tf.layers.dropout(encoding, rate=drop_prob)
        for i in range(num_layers - 1):
            encoding = tf.layers.dense(encoding, units, activation=tf.nn.tanh)
            # encoding = tf.layers.dropout(encoding, rate=drop_prob)

        w = tf.get_variable('w', [units, output_units])
        b = tf.get_variable('b', [output_units])
        latent_encoding = tf.nn.xw_plus_b(encoding, w, b)

        zeros = tf.zeros([batch_size, output_units])
        condition = tf.not_equal(inputs, 0)

        self.latent_encoding = tf.where(condition, latent_encoding, zeros)

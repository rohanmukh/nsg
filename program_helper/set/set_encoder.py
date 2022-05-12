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


class KeywordEncoder(object):

    def __init__(self, keywords, units, num_layers, vocab_size, batch_size, max_kw_size,
                 emb=None,
                 drop_prob=None):

        if drop_prob is None:
            drop_prob = tf.constant(0.0, dtype=tf.float32)

        '''
        Initial keywords have shape (batch_size, max_kw_size)
        Lets drop a few
        '''
        dropper = tf.cast(tf.random_uniform((batch_size, max_kw_size), 0, 1, dtype=tf.float32) > drop_prob, tf.int32)
        keywords = keywords * dropper

        '''
        now we will reshape it so that the same network can be applied to all evidences
        '''
        keywords = tf.reshape(keywords, [batch_size * max_kw_size])

        keywords_emb = tf.nn.embedding_lookup(emb, keywords)

        layer1 = tf.layers.dense(keywords_emb, units, activation=tf.nn.tanh)
        # layer1 = tf.layers.dropout(layer1, rate=drop_prob)
        for i in range(num_layers - 1):
            layer1 = tf.layers.dense(layer1, units, activation=tf.nn.tanh)
            # layer1 = tf.layers.dropout(layer1, rate=drop_prob)

        layer2 = tf.layers.dense(layer1, units)  # [bs*max_kw, units]
        # layer2 = tf.layers.dropout(layer2, rate=drop_prob)

        non_zeros = tf.where(tf.not_equal(keywords, 0), layer2, tf.zeros_like(layer2))
        reshaper = tf.reshape(non_zeros, [batch_size, max_kw_size, units])

        self.output = tf.reduce_sum(reshaper, axis=1)
        return

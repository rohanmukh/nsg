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

from program_helper.element.dense_encoder import DenseEncoder
from program_helper.sequence.sequence_encoder import SequenceEncoder
from program_helper.set.set_encoder import KeywordEncoder


class SurroundingEncoder(object):

    def __init__(self, surr_ret, surr_fp, surr_method,
                 units, num_layers,
                 type_vocab_size, kw_vocab_size,
                 batch_size,
                 max_kws,
                 max_camel_case,
                 type_emb,
                 kw_emb,
                 drop_prob=None
                 ):

        '''
            Note that none of the individual components have drop prob, but the methods are dropped as a whole
        '''
        ## Before: BS*max_kw, After: (BS*max_kw)*1
        with tf.variable_scope("return_type"):
            surr_ret = tf.reshape(surr_ret, [-1])
            ret_mean_enc = DenseEncoder(surr_ret,
                                        units, num_layers,
                                        units,
                                        type_vocab_size,
                                        batch_size * max_kws,
                                        emb=type_emb,
                                        # drop_prob=drop_prob
                                        )

            ret_mean_enc = tf.reshape(ret_mean_enc.latent_encoding, [batch_size, max_kws, -1])

        ## Before: BS*max_kw*max_cc, After: (BS*max_kw)*max_cc
        with tf.variable_scope("formal_param"):
            surr_fp = tf.reshape(surr_fp, [-1, max_camel_case])
            fp_mean_enc = KeywordEncoder(surr_fp,
                                          units, num_layers,
                                          type_vocab_size,
                                          batch_size * max_kws,
                                          max_camel_case,
                                          emb=type_emb,
                                          # drop_prob=drop_prob
                                          )
            fp_mean_enc = tf.reshape(fp_mean_enc.output, [batch_size, max_kws, -1])

        ## Before: BS*max_kw*max_cc, After: (BS*max_kw)*max_cc
        with tf.variable_scope("method"):
            surr_method = tf.reshape(surr_method, [-1, max_camel_case])
            method_mean_enc = KeywordEncoder(surr_method,
                                              units, num_layers,
                                              kw_vocab_size,
                                              batch_size * max_kws,
                                              max_camel_case,
                                              emb=kw_emb,
                                              # drop_prob=drop_prob
                                              )
            method_mean_enc = tf.reshape(method_mean_enc.output, [batch_size, max_kws, -1])

        merged_mean = tf.concat([ret_mean_enc, fp_mean_enc, method_mean_enc], axis=2)
        '''
            merged mean is batch_size * num_methods * dimension
        '''

        layer1 = tf.layers.dense(merged_mean, units, activation=tf.nn.tanh)
        # layer1 = tf.layers.dropout(layer1, rate=drop_prob)

        layer2 = tf.layers.dense(layer1, units, activation=tf.nn.tanh)
        layer3 = tf.layers.dense(layer2, units)
        '''
            layer3 is batch_size * num_methods * dimension
            lets drop  a few methods
        '''

        self.internal_method_embedding = tf.where( tf.greater(ret_mean_enc, 0.), layer3, tf.zeros_like(layer3))
        dropper = tf.cast(tf.random_uniform((batch_size, max_kws), 0, 1, dtype=tf.float32) > drop_prob, tf.float32)
        self.dropped_embedding = dropper[:, :, None] * self.internal_method_embedding

        self.output = tf.reduce_sum(self.dropped_embedding, axis=1)

        return

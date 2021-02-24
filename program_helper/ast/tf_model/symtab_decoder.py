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
from program_helper.sequence.base_lstm_class import BaseLSTMClass


class SymTabDecoder(BaseLSTMClass):
    def __init__(self, units, num_layers,
                 vocab_size=None,
                 num_vars=None,
                 batch_size=None,
                 drop_prob=None):
        super().__init__(units, 1, num_vars, drop_prob)

        self.num_vars = num_vars
        # CELL 1 is to update symbol table
        self.cell1 = self.create_lstm_cell()

        self.input_embeddings = self.create_input_embedding(vocab_size)

        return

    def decoder(self, input, state_in, inp_symtab):
        # temp = tf.reshape(inp_symtab, (-1, self.num_vars, self.units))
        emb_inp = tf.reshape(tf.tile(input, (1, self.num_vars)), (-1, self.num_vars, self.units))

        concatted = tf.concat([inp_symtab, emb_inp], axis=2)
        l1 = tf.layers.dense(concatted, self.units, activation=tf.nn.tanh)
        l1_flat = tf.reshape(l1, (-1, self.num_vars * self.units))
        l2 = tf.layers.dense( l1_flat, self.num_vars, activation=tf.nn.tanh)
        l3 = tf.layers.dense(l2, self.num_vars)

        return l3, state_in

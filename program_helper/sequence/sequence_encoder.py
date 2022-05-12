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
from program_helper.sequence.simple_lstm import SimpleLSTM


class SequenceEncoder(SimpleLSTM):
    def __init__(self, inputs, units, num_layers, batch_size, output_units, type_emb, drop_prob=None):

        super().__init__(units, num_layers, output_units, drop_prob)
        self.type_emb = type_emb



        with tf.variable_scope('encoder_network'):
            self.state = self.get_initial_state(batch_size)
            output = tf.zeros((batch_size, units))
            for i, inp in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                emb_inp = tf.nn.embedding_lookup(self.type_emb, inp)
                curr_out, out_state = self.get_next_output(emb_inp, self.state)

                '''
                lets drop it randomly
                '''
                dropper = tf.cast(tf.random_uniform((batch_size, 1), 0, 1, dtype=tf.float32)
                                  > drop_prob, tf.int32)
                inp = inp * tf.squeeze(dropper)
                cond = tf.not_equal(inp, 0)

                self.state = [tf.where(cond, out_state[k], self.state[k]) for k in range(num_layers)]
                output = tf.where(cond, curr_out, output)

        self.last_output = self.get_projection(output)

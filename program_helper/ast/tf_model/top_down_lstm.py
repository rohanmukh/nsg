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


class TopDownLSTM(SimpleLSTM):
    def __init__(self, units, num_layers,
                 output_units=None,
                 drop_prob=None):

        super().__init__(units, num_layers, output_units, drop_prob)

        self.cell2 = self.create_lstm_cell() #extra LSTM for child edges

        return

    def get_next_output_with_symtab(self, mod_input, edge, curr_state):

        with tf.variable_scope('cell1'):  # handles CHILD_EDGE
            output1, state1 = self.cell1(mod_input, curr_state)
        with tf.variable_scope('cell2'):  # handles SIBLING EDGE
            output2, state2 = self.cell2(mod_input, curr_state)

        output = tf.where(edge, output1, output2)
        state = [tf.where(edge, state1[j], state2[j]) for j in range(self.num_layers)]

        return output, state
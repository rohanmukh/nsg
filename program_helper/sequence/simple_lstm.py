
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


class SimpleLSTM(BaseLSTMClass):
    def __init__(self, units, num_layers,
                 output_units=None, drop_prob=None):
        super().__init__(units, num_layers,
                         output_units=output_units,
                         drop_prob=drop_prob)

        self.cell1 = self.create_lstm_cell()
        self.projection_w, self.projection_b = self.create_projections()
        return

    def get_projection(self, input):
        return tf.nn.xw_plus_b(input, self.projection_w, self.projection_b)

    def get_next_output(self, emb_inp, curr_state):
        output, out_state = self.cell1(emb_inp, curr_state)

        curr_state = [curr_state[j]
                      for j in range(self.num_layers)]

        return output, curr_state

    def get_next_output_with_symtab(self, mod_input, curr_state):

        output, out_state = self.cell1(mod_input, curr_state)

        return output, out_state

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


class BaseLSTMClass:
    def __init__(self, units, num_layers,
                 output_units=None,
                 drop_prob=None):

        if drop_prob is None:
            self.drop_prob = tf.placeholder_with_default(1., shape=())
        else:
            self.drop_prob = drop_prob

        self.units = units
        self.num_layers = num_layers
        self.output_units = output_units

        return


    def get_initial_state(self, batch_size):
        # initial_state has get_shape (batch_size, latent_size), same as psi_mean in the prev code
        init_state = [tf.random.truncated_normal([batch_size, self.units],
                                                 stddev=0.001)] * self.num_layers
        # curr_out = tf.zeros([batch_size, units])
        return init_state

    def create_lstm_cell(self):
        cells1 = []
        for _ in range(self.num_layers):
            cell1 = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(self.units)
            cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1,
                                                  state_keep_prob=self.drop_prob)
            cells1.append(cell1)

        cell1 = tf.nn.rnn_cell.MultiRNNCell(cells1)

        return cell1

    def create_projections(self):
        # projection matrices for output
        projection_w = tf.get_variable('projection_w', [self.units, self.output_units])
        projection_b = tf.get_variable('projection_b', [self.output_units])
        return projection_w, projection_b





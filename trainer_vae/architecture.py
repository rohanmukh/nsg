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
from program_helper.program_encoder import ProgramEncoder
from program_helper.program_decoder import ProgramDecoder


class Encoder(object):

    def __init__(self, config,
                 fp_inputs, field_inputs,
                 apicalls, types, keywords,
                 ret,
                 method, classname, javadoc,
                 surr_ret, surr_fp, surr_method
                 ):
        self.ev_drop_rate = tf.squeeze(tf.placeholder_with_default(tf.constant(0.0, shape=(1, 10),
                                                                            dtype=tf.float32), (1, 10)))
        self.ev_miss_rate = tf.squeeze(tf.placeholder_with_default(tf.constant(0.0, shape=(1, 10),
                                                                               dtype=tf.float32), (1, 10)))
        with tf.variable_scope("Mean"):
            self.program_encoder = ProgramEncoder(config,
                                                  apicalls, types, keywords,
                                                  fp_inputs, field_inputs,
                                                  ret,
                                                  method, classname, javadoc,
                                                  surr_ret, surr_fp, surr_method,
                                                  ev_drop_rate=self.ev_drop_rate,
                                                  ev_miss_rate=self.ev_miss_rate
                                                  )
            self.output_mean = self.program_encoder.mean
            self.output_covar = self.program_encoder.covar


class Decoder(object):
    def __init__(self, config,
                 nodes, edges, var_decl_ids, ret_reached,
                 iattrib, all_var_mappers,
                 type_helper_val, expr_type_val, ret_type_val,
                 node_type_number,
                 fp_inputs,
                 field_inputs,
                 ret_type, method_embedding,
                 initial_state,
                 gnn_inputs=None):
        self.program_decoder = ProgramDecoder(config, nodes, edges,
                                              var_decl_ids, ret_reached,
                                              iattrib, all_var_mappers,
                                              type_helper_val, expr_type_val, ret_type_val,
                                              node_type_number,
                                              fp_inputs, field_inputs,
                                              ret_type, method_embedding, initial_state,
                                              gnn_inputs)
        self.ast_logits = self.program_decoder.ast_tree.output_logits

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

from program_helper.ast.tf_model.base_ast_encoder import BaseTreeEncoding

MAX_VARIABLES = 10


class TreeEncoder(BaseTreeEncoding):
    def __init__(self, nodes, edges, var_decl_ids,
                 type_helper_val, expr_type_val, ret_type_val,
                 varOrNot, typeOrNot, apiOrNot, symtabmod_or_not, op_or_not,
                 fp_inputs, ret_type,
                 units, num_layers, output_units, batch_size,
                 api_vocab_size, type_vocab_size, var_vocab_size, concept_vocab_size, op_vocab_size,
                 type_emb, concept_emb, api_emb, var_emb, op_emb,
                 drop_prob=None):

        super().__init__(units, num_layers, output_units, batch_size,
                         api_vocab_size, type_vocab_size, var_vocab_size, concept_vocab_size, op_vocab_size,
                         type_emb, concept_emb, api_emb, var_emb, op_emb,
                         drop_prob)

        self.init_symtab = self.symtab_encoder.create_symtab(batch_size, MAX_VARIABLES, units)
        method_ret_type_emb = tf.expand_dims(tf.nn.embedding_lookup(type_emb, ret_type), axis=1)
        method_fp_type_emb = tf.stack([tf.nn.embedding_lookup(type_emb, fp_type) for fp_type in fp_inputs], axis=1)

        with tf.variable_scope('tree_encoder'):

            self.state = self.get_initial_state(batch_size)
            self.symtab = self.init_symtab
            output = tf.zeros((self.batch_size, self.units), dtype=tf.float32)

            for i in range(len(nodes)):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()

                self.state, output, self.symtab, _ = self.encode(nodes[i], edges[i],
                                                                          var_decl_ids[i],
                                                                          type_helper_val[i], expr_type_val[i],
                                                                          ret_type_val[i],
                                                                          varOrNot[i], typeOrNot[i],
                                                                          apiOrNot[i], symtabmod_or_not[i],
                                                                          op_or_not[i],
                                                                          self.symtab,
                                                                          method_ret_type_emb, method_fp_type_emb,
                                                                          self.state,
                                                                          )

        self.last_output = self.get_projection(output)


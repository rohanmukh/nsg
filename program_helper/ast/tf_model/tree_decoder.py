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
from utilities.tensor_permutor import permute_batched_tensor_3dim


class TreeDecoder(BaseTreeEncoding):
    def __init__(self, nodes, edges,
                 var_decl_ids, ret_reached, iattrib,  all_var_mappers,
                 type_helper_val, expr_type_val, ret_type_val,
                 node_type_number,
                 ret_type, fps,  field_inputs,
                 initial_state, method_embedding,
                 num_layers, units, batch_size,
                 api_vocab_size, type_vocab_size,
                 var_vocab_size, concept_vocab_size, op_vocab_size, method_vocab_size,
                 type_emb, concept_emb,
                 drop_prob=None,
                 max_variables=None,
                 ):

        self.drop_prob = drop_prob

        super().__init__(units, num_layers, units, batch_size,
                         api_vocab_size, type_vocab_size, var_vocab_size,
                         concept_vocab_size, op_vocab_size, method_vocab_size,
                         type_emb, concept_emb,
                         max_variables=max_variables,
                         drop_prob=drop_prob)

        self.init_symtab = self.symtab_encoder.create_symtab(batch_size, units)
        self.init_unused_varflag = self.symtab_encoder.create_unused_varflag(batch_size)
        self.init_nullptr_varflag = self.symtab_encoder.create_nullptr_varflag(batch_size)


        method_ret_type_emb = tf.expand_dims(tf.nn.embedding_lookup(type_emb, ret_type), axis=1)

        ## FP symtab
        method_fp_type_emb = tf.stack([tf.nn.embedding_lookup(type_emb, fp_type) for fp_type in fps], axis=1)
        method_fp_type_emb = permute_batched_tensor_3dim(method_fp_type_emb, all_var_mappers[:, 10:20])

        ## Field symtab
        method_field_type_emb = tf.stack([tf.nn.embedding_lookup(type_emb, fp_type) for fp_type in field_inputs], axis=1)
        method_field_type_emb = permute_batched_tensor_3dim(method_field_type_emb, all_var_mappers[:, 20:])

        ## Var symtab mappers
        internal_var_mapper = all_var_mappers[:, :10]

        internal_method_embedding = method_embedding

        with tf.variable_scope('tree_decoder'):
            self.state = initial_state
            self.symtab = self.init_symtab
            self.unused_varflag = self.init_unused_varflag
            self.nullptr_varflag = self.init_nullptr_varflag

            api_output_logits, type_output_logits, clstype_output_logits, \
            var_output_logits, concept_output_logits,\
                op_output_logits, method_output_logits = [], [], [], [], [], [], []
            for i in range(len(nodes)):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()

                self.state, self.symtab, self.unused_varflag, self.nullptr_varflag, logits = \
                                                   self.get_next_output(nodes[i], edges[i],
                                                   var_decl_ids[i], ret_reached[i], iattrib[i],
                                                   type_helper_val[i], expr_type_val[i],
                                                   ret_type_val[i],
                                                   node_type_number[i],
                                                   self.symtab,
                                                   self.unused_varflag,
                                                   self.nullptr_varflag,
                                                   method_ret_type_emb, method_fp_type_emb,
                                                   method_field_type_emb,
                                                   internal_method_embedding,
                                                   self.state,
                                                   internal_var_mapper
                                           )

                api_logit, type_logit, clstype_logit, var_logit, concept_logit, op_logit, method_logit = logits

                api_output_logits.append(api_logit)
                type_output_logits.append(type_logit)
                clstype_output_logits.append(clstype_logit)
                var_output_logits.append(var_logit)
                concept_output_logits.append(concept_logit)
                op_output_logits.append(op_logit)
                method_output_logits.append(method_logit)

        self.output_logits = [
            tf.stack(concept_output_logits, 1),
            tf.stack(api_output_logits, 1),
            tf.stack(type_output_logits, 1),
            tf.stack(clstype_output_logits, 1),
            tf.stack(var_output_logits, 1),
            tf.stack(op_output_logits, 1),
            tf.stack(method_output_logits, 1),
            tf.ones((batch_size, len(nodes), batch_size), dtype=tf.float32)
        ]

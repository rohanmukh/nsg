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

from program_helper.ast.tf_model.top_down_lstm import TopDownLSTM
from program_helper.ast.tf_model.symtab_encoder import SymTabEncoder
from program_helper.sequence.simple_lstm import BaseLSTMClass, SimpleLSTM
from synthesis.ops.candidate_ast import SYMTAB_MOD, VAR_NODE, TYPE_NODE, API_NODE, OP_NODE, METHOD_NODE, CLSTYPE_NODE
from utilities.tensor_permutor import permute_batched_tensor_3dim, permute_batched_tensor_2dim


class BaseTreeEncoding(BaseLSTMClass):
    def __init__(self, units, num_layers, output_units, batch_size,
                 api_vocab_size, type_vocab_size, var_vocab_size,
                 concept_vocab_size, op_vocab_size, method_vocab_size,
                 type_emb, concept_emb,
                 max_variables=None,
                 drop_prob=None):
        super().__init__(units, num_layers, output_units, drop_prob)
        self.projection_w, self.projection_b = self.create_projections()

        self.type_emb = type_emb
        self.concept_emb = concept_emb

        self.max_variables = max_variables

        self.batch_size = batch_size
        self.initialize_lstms(units, num_layers, concept_vocab_size, type_vocab_size,
                              api_vocab_size,
                              var_vocab_size, op_vocab_size, method_vocab_size)

    def initialize_lstms(self, units, num_layers, concept_vocab_size, type_vocab_size,
                         api_vocab_size, var_vocab_size, op_vocab_size, method_vocab_size):
        with tf.variable_scope('concept_prediction'):
            self.concept_encoder = TopDownLSTM(units, num_layers,
                                               output_units=concept_vocab_size)

        with tf.variable_scope('api_prediction'):
            self.api_encoder = SimpleLSTM(units, num_layers,
                                          output_units=api_vocab_size)

        with tf.variable_scope('type_prediction'):
            self.type_encoder = SimpleLSTM(units, num_layers,
                                           output_units=type_vocab_size)

        with tf.variable_scope('clstype_prediction'):
            self.clstype_encoder = SimpleLSTM(units, num_layers,
                                           output_units=type_vocab_size)

        with tf.variable_scope('var_access_prediction_type'):
            self.var_encoder1 = SimpleLSTM(units, num_layers,
                                           output_units=var_vocab_size)

        with tf.variable_scope('var_access_prediction_exp'):
            self.var_encoder2 = SimpleLSTM(units, num_layers,
                                           output_units=var_vocab_size)

        with tf.variable_scope('var_access_prediction_ret'):
            self.var_encoder3 = SimpleLSTM(units, num_layers,
                                           output_units=var_vocab_size)

        with tf.variable_scope('var_declaration'):
            self.var_encoder4 = SimpleLSTM(units, num_layers,
                                           output_units=var_vocab_size)


        with tf.variable_scope('op_prediction'):
            self.op_encoder = SimpleLSTM(units, num_layers,
                                           output_units=op_vocab_size)

        with tf.variable_scope('method_prediction'):
            self.method_encoder = SimpleLSTM(units, num_layers,
                                           output_units=method_vocab_size)


        with tf.variable_scope("symtab_updater"):
            self.symtab_encoder = SymTabEncoder(units, num_layers,
                                                num_vars=self.max_variables,
                                                batch_size=self.batch_size)

        return

    def get_next_output(self, node, edge,
                        var_decl_id, ret_reached, iattrib,
                        type_helper_val, expr_type_val, ret_type_val,
                        node_type_number,
                        symtab_in, unused_varflag_in, nullptr_varflag_in,
                        method_ret_type_helper, method_fp_type_emb,
                        method_field_type_emb, internal_method_embedding,
                        state_in, internal_var_mapper):

        method_ret_type_helper = method_ret_type_helper[:,0,:]
        # Var declaration ID is decremented by 1. This is because when input var decl id is 1
        # or when we see the first variable, we tag it in symtab as 0-th var
        # When there are no variable, var_decl_id becomes -1 and does not update the symtab
        var_decl_id = var_decl_id - 1

        with tf.variable_scope('symtab_in'):
            ## For unused varflags lets permute them
            flat_varflag = permute_batched_tensor_2dim(unused_varflag_in, internal_var_mapper)
            flat_nullptr_varflag_in = permute_batched_tensor_2dim(nullptr_varflag_in, internal_var_mapper)

            ## Easy to process the ret_reached attribute
            ret_reached = tf.expand_dims(tf.cast(ret_reached, tf.float32), axis=1)
            iattrib = tf.cast(iattrib, tf.float32)

            ## For input vars symtab, first permute them and then learn using a few layers
            perm_inp_vars = permute_batched_tensor_3dim(symtab_in, internal_var_mapper)
            symtab_all = tf.concat([perm_inp_vars,
                                    method_fp_type_emb, method_field_type_emb], axis=1)
            symtab_all = tf.layers.dense(symtab_all, self.units, activation=tf.nn.tanh)
            symtab_all = tf.layers.dense(symtab_all, self.units)
            flat_symtab = tf.reshape(symtab_all, (self.batch_size, -1))

        with tf.variable_scope('concept_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               flat_varflag,
                               flat_nullptr_varflag_in,
                               ret_reached,
                               iattrib,
                               method_ret_type_helper,
                               flat_symtab,
                               ], axis=1)
            concept_output, concept_state = self.concept_encoder.get_next_output_with_symtab(input, edge,
                                                                                             state_in)
            concept_logit = self.concept_encoder.get_projection(concept_output)

        with tf.variable_scope('api_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               flat_varflag,
                               flat_nullptr_varflag_in,
                               ret_reached,
                               method_ret_type_helper,
                               iattrib
                               ], axis=1)
            api_output, api_state = self.api_encoder.get_next_output_with_symtab(input, state_in)
            api_logit = self.api_encoder.get_projection(api_output)

        with tf.variable_scope('type_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               flat_varflag,
                               flat_nullptr_varflag_in,
                               ret_reached,
                               method_ret_type_helper,
                               iattrib
                               ], axis=1)
            type_output, type_state = self.type_encoder.get_next_output_with_symtab(input, state_in)
            type_logit = self.type_encoder.get_projection(type_output)

        with tf.variable_scope('clstype_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               flat_varflag,
                               flat_nullptr_varflag_in,
                               ret_reached,
                               method_ret_type_helper,
                               iattrib
                               ], axis=1)
            clstype_output, clstype_state = self.clstype_encoder.get_next_output_with_symtab(input, state_in)
            clstype_logit = self.clstype_encoder.get_projection(clstype_output)

        with tf.variable_scope('op_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               flat_varflag,
                               flat_nullptr_varflag_in,
                               ret_reached,
                               iattrib
                               ], axis=1)
            op_output, op_state = self.op_encoder.get_next_output_with_symtab(input, state_in)
            op_logit = self.op_encoder.get_projection(op_output)

        with tf.variable_scope('method_prediction'):
            internal_method_embedding_flat = tf.reshape(internal_method_embedding, (self.batch_size, -1))
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               internal_method_embedding_flat,
                               flat_varflag,
                               flat_nullptr_varflag_in,
                               ret_reached,
                                ], axis=1)
            method_output, method_state = self.method_encoder.get_next_output_with_symtab(input, state_in)
            method_logit = self.method_encoder.get_projection(method_output)

        with tf.variable_scope('var_access_prediction_type'):
            input1 = tf.concat([flat_symtab,
                                flat_varflag,
                                flat_nullptr_varflag_in,
                                tf.nn.embedding_lookup(self.type_emb, type_helper_val),
                                ], axis=1)
            input1 = tf.layers.dense(input1, self.units, activation=tf.nn.tanh)
            input1 = tf.layers.dense(input1, self.units, activation=tf.nn.tanh)
            var_output1, var_state1 = self.var_encoder1.get_next_output_with_symtab(input1, state_in)

        with tf.variable_scope('var_access_prediction_exp'):
            input2 = tf.concat([flat_symtab,
                                flat_varflag,
                                flat_nullptr_varflag_in,
                                tf.nn.embedding_lookup(self.type_emb, expr_type_val),
                                ], axis=1)
            input2 = tf.layers.dense(input2, self.units, activation=tf.nn.tanh)
            input2 = tf.layers.dense(input2, self.units, activation=tf.nn.tanh)
            var_output2, var_state2 = self.var_encoder2.get_next_output_with_symtab(input2, state_in)

        with tf.variable_scope('var_access_prediction_ret'):
            input3 = tf.concat([flat_symtab,
                                flat_varflag,
                                flat_nullptr_varflag_in,
                                tf.nn.embedding_lookup(self.type_emb, ret_type_val),
                                ], axis=1)
            input3 = tf.layers.dense(input3, self.units, activation=tf.nn.tanh)
            input3 = tf.layers.dense(input3, self.units, activation=tf.nn.tanh)
            var_output3, var_state3 = self.var_encoder3.get_next_output_with_symtab(input3, state_in)

        with tf.variable_scope('var_declaration'):
            input4 = tf.concat([flat_symtab,
                                flat_varflag,
                                flat_nullptr_varflag_in,
                                ], axis=1)
            input4 = tf.layers.dense(input4, self.units, activation=tf.nn.tanh)
            input4 = tf.layers.dense(input4, self.units, activation=tf.nn.tanh)
            var_output4, var_state4 = self.var_encoder4.get_next_output_with_symtab(input4, state_in)


        var_state = [tf.where(tf.not_equal(type_helper_val, 0),
                              var_state1[j],
                              tf.where(tf.not_equal(expr_type_val, 0),
                                       var_state2[j],
                                       tf.where(tf.not_equal(ret_type_val, 0),
                                                var_state3[j],
                                                var_state4[j]
                                                )
                                       )
                              )
                     for j in range(self.num_layers)]

        var_logit = tf.where(tf.not_equal(type_helper_val, 0),
                             self.var_encoder1.get_projection(var_output1),
                             tf.where(tf.not_equal(expr_type_val, 0),
                                      self.var_encoder2.get_projection(var_output2),
                                      tf.where(tf.not_equal(ret_type_val, 0),
                                               self.var_encoder3.get_projection(var_output3),
                                               self.var_encoder3.get_projection(var_output4)
                                               )
                                      )
                             )


        # or nots
        symtabmod_or_not = tf.equal(node_type_number, SYMTAB_MOD)
        var_or_not = tf.equal(node_type_number, VAR_NODE)
        type_or_not = tf.equal(node_type_number, TYPE_NODE)
        clstype_or_not = tf.equal(node_type_number, CLSTYPE_NODE)
        api_or_not = tf.equal(node_type_number, API_NODE)
        op_or_not = tf.equal(node_type_number, OP_NODE)
        method_or_not = tf.equal(node_type_number, METHOD_NODE)


        # Update symtab
        with tf.variable_scope("symtab_updater"):
            '''
                update symtab first.
            '''
            input = tf.nn.embedding_lookup(self.type_emb, type_helper_val)
            new_symtab = self.symtab_encoder.update_symtab(input, var_decl_id, symtab_in)
            new_symtab = tf.where(symtabmod_or_not, new_symtab, symtab_in)
            stripped_symtab = self.symtab_encoder.strip_symtab(var_decl_id, new_symtab)

            '''
                next we update the attribute: unused var flag
                type_or_not or clstype_or_not gives the hint correctly when DVarDecl is defined
            '''
            new_unused_vars_decl = self.symtab_encoder.decl_update_unused_vars(var_decl_id, unused_varflag_in)
            new_unused_vars_invoke = self.symtab_encoder.usage_update_unused_vars(var_decl_id, unused_varflag_in)
            new_unused_vars = tf.where(tf.math.logical_or(type_or_not, clstype_or_not),
                                       new_unused_vars_decl,
                                       tf.where(var_or_not,
                                                new_unused_vars_invoke,
                                                unused_varflag_in
                                                )
                                       )
            # Might not need to strip this
            stripped_new_unused_vars = self.symtab_encoder.strip_unused_vars(var_decl_id, new_unused_vars)

            '''
                next we update the attribute: unused var flag
                type_or_not or clstype_or_not gives the hint correctly when DVarDecl is defined
            '''
            new_nullptr_decl = self.symtab_encoder.decl_update_nullptr_varflag(var_decl_id, nullptr_varflag_in)
            new_nullpte_invoke = self.symtab_encoder.usage_update_nullptr_varflag(var_decl_id, nullptr_varflag_in)
            new_nullptr_varflag = tf.where(clstype_or_not,
                                       new_nullptr_decl,
                                       tf.where(var_or_not,
                                                new_nullpte_invoke,
                                                nullptr_varflag_in
                                                )
                                       )
            # Might not need to strip this
            stripped_new_nullptr_varflag = self.symtab_encoder.strip_nullptr_varflag(var_decl_id, new_nullptr_varflag)


        # Update state
        state = [tf.where(symtabmod_or_not, state_in[j], tf.where(var_or_not,
                              var_state[j],
                              tf.where(type_or_not,
                                       type_state[j],
                                       tf.where(
                                           api_or_not,
                                           api_state[j],
                                           tf.where(op_or_not,
                                                    op_state[j],
                                                    tf.where(method_or_not,
                                                             method_state[j],
                                                             tf.where(clstype_or_not,
                                                                      clstype_state[j],
                                                                      concept_state[j])
                                                            )
                                                    )
                                           )
                                       )
                              )
                          )
                 for j in range(self.num_layers)]

        logits = [api_logit, type_logit, clstype_logit,
                  var_logit, concept_logit, op_logit,
                  method_logit]

        return state, stripped_symtab, stripped_new_unused_vars, stripped_new_nullptr_varflag, logits


    def get_projection(self, input):
        return tf.nn.xw_plus_b(input, self.projection_w, self.projection_b)


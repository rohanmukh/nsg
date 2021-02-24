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
import numpy as np
import tensorflow as tf

from trainer_vae.architecture import Encoder, Decoder
from tensorflow.contrib import seq2seq
from synthesis.ops.candidate_ast import TYPE_NODE, VAR_NODE, API_NODE, SYMTAB_MOD, OP_NODE, CONCEPT_NODE, METHOD_NODE, \
    CLSTYPE_NODE, VAR_DECL_NODE


class Model:
    def __init__(self, config, top_k=5):
        self.config = config

        self.nodes = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])
        self.edges = tf.placeholder(tf.bool, [self.config.batch_size, self.config.max_ast_depth])
        self.targets = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])

        self.var_decl_ids = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])
        self.ret_reached = tf.placeholder(tf.bool, [self.config.batch_size, self.config.max_ast_depth])
        self.iattrib = tf.placeholder(tf.bool, [self.config.batch_size, self.config.max_ast_depth, 3])

        self.all_var_mappers = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_variables
                                                         + self.config.input_fp_depth + self.config.max_fields])

        self.node_type_number = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])

        self.type_helper_val = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])
        self.expr_type_val = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])
        self.ret_type_val = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_ast_depth])

        self.return_type = tf.placeholder(tf.int32, [self.config.batch_size])

        self.formal_param_inputs = tf.placeholder(tf.int32, [self.config.batch_size, self.config.input_fp_depth])
        self.field_inputs = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_fields])

        self.apicalls = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_keywords])
        self.types = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_keywords])
        self.keywords = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_keywords])

        self.method = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_camel_case])
        self.classname = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_camel_case])
        self.javadoc_kws = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_keywords])

        self.surr_ret = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_keywords])
        self.surr_fp = tf.placeholder(tf.int32,
                                      [self.config.batch_size, self.config.max_keywords, self.config.max_camel_case])
        self.surr_method = tf.placeholder(tf.int32, [self.config.batch_size, self.config.max_keywords,
                                                     self.config.max_camel_case])

        nodes = tf.unstack(tf.transpose(self.nodes), axis=0)
        edges = tf.unstack(tf.transpose(self.edges), axis=0)
        # targets = tf.unstack(tf.transpose(self.targets), axis=0)

        var_decl_ids = tf.unstack(tf.transpose(self.var_decl_ids), axis=0)
        ret_reached = tf.unstack(tf.transpose(self.ret_reached), axis=0)
        iattrib = tf.unstack(self.iattrib, axis=1)

        node_type_number = tf.unstack(tf.transpose(self.node_type_number), axis=0)

        type_helper_val = tf.unstack(tf.transpose(self.type_helper_val), axis=0)
        expr_type_val = tf.unstack(tf.transpose(self.expr_type_val), axis=0)
        ret_type_val = tf.unstack(tf.transpose(self.ret_type_val), axis=0)

        formal_param_inputs = tf.unstack(tf.transpose(self.formal_param_inputs), axis=0)
        field_inputs = tf.unstack(tf.transpose(self.field_inputs), axis=0)
        self.latent_state = tf.random.normal([config.batch_size, config.latent_size], mean=0., stddev=1.,
                                             dtype=tf.float32)

        with tf.variable_scope("encoder"):
            self.encoder = Encoder(config, self.formal_param_inputs, self.field_inputs,
                                   self.apicalls, self.types, self.keywords,
                                   self.return_type, self.method, self.classname, self.javadoc_kws,
                                   self.surr_ret, self.surr_fp, self.surr_method)
            samples = tf.random.normal([config.batch_size, config.latent_size], mean=0., stddev=1.,
                                       dtype=tf.float32)
            self.latent_state = self.encoder.output_mean + tf.sqrt(self.encoder.output_covar) * samples

        # 2. KL loss: negative of the KL-divergence between P(\Psi | f(\Theta)) and P(\Psi)
        self.KL_loss = tf.reduce_mean(0.5 * tf.reduce_sum(- tf.math.log(self.encoder.output_covar)
                                                          - 1 + self.encoder.output_covar
                                                          + tf.square(-self.encoder.output_mean)
                                                          , axis=1), axis=0)

        with tf.variable_scope("decoder"):
            latent_state_lifted = tf.layers.dense(self.latent_state, config.decoder.units)
            self.initial_state = [latent_state_lifted] * config.decoder.num_layers
            self.decoder = Decoder(config, nodes, edges,
                                   var_decl_ids, ret_reached, iattrib,
                                   self.all_var_mappers,
                                   type_helper_val, expr_type_val, ret_type_val,
                                   node_type_number,
                                   formal_param_inputs, field_inputs,
                                   self.return_type,
                                   self.encoder.program_encoder.surr_enc.internal_method_embedding,
                                   self.initial_state)

        def get_loss(id, node_type):
            weights = tf.ones_like(self.targets, dtype=tf.float32) \
                      * tf.cast(tf.greater(self.targets, 0), tf.float32) \
                      * tf.cast(tf.equal(self.node_type_number, node_type), tf.float32)

            targets = self.targets \
                      * tf.cast(tf.equal(self.node_type_number, node_type), tf.int32)

            loss_matrix = seq2seq.sequence_loss(self.decoder.ast_logits[id], targets,
                                      weights,
                                      average_across_batch=False,
                                      average_across_timesteps=False)
            prob = tf.reduce_mean(tf.reduce_sum( loss_matrix
                , axis=1), axis=0)
            return loss_matrix, prob

        ############################
        # 1d. Generator loss for AST Concept calls
        self.ast_gen_loss_concept_matrix, self.ast_gen_loss_concept = get_loss(0, CONCEPT_NODE)
        self.ast_gen_loss_api_matrix, self.ast_gen_loss_api = get_loss(1, API_NODE)
        self.ast_gen_loss_type_matrix, self.ast_gen_loss_type = get_loss(2, TYPE_NODE)
        self.ast_gen_loss_clstype_matrix, self.ast_gen_loss_clstype = get_loss(3, CLSTYPE_NODE)
        self.ast_gen_loss_var_matrix, self.ast_gen_loss_var = get_loss(4, VAR_NODE)
        self.ast_gen_loss_vardecl_matrix, self.ast_gen_loss_vardecl = get_loss(5, VAR_DECL_NODE)
        # self.ast_gen_loss_op_matrix, self.ast_gen_loss_op = get_loss(6, OP_NODE)
        self.ast_gen_loss_method_matrix, self.ast_gen_loss_method = get_loss(7, METHOD_NODE)
        ############################

        self.ast_gen_loss = self.ast_gen_loss_method +  \
                            self.ast_gen_loss_concept + self.ast_gen_loss_api \
                            + self.ast_gen_loss_clstype + self.ast_gen_loss_type \
                            + self.ast_gen_loss_var + self.ast_gen_loss_vardecl \
                            # self.ast_gen_loss_op +

        with tf.name_scope("ast_inference"):
            ast_logits = [self.decoder.ast_logits[i][:, 0, :] for i in range(len(self.decoder.ast_logits))]
            self.ast_ln_probs = [tf.nn.log_softmax(temp) for temp in ast_logits]
            self.ast_idx = [tf.multinomial(temp, 1) for temp in self.ast_ln_probs]
            self.ast_top_k_values, self.ast_top_k_indices = [], []
            for temp in self.ast_ln_probs:
                beam_width = temp.shape[0]
                max_vocab_sz = temp.shape[1]
                mod_top_k = np.minimum(max_vocab_sz, top_k)
                vals, indices = tf.nn.top_k(temp, k=mod_top_k)

                const = tf.constant(-20000.0, shape=(beam_width, top_k - mod_top_k), dtype=tf.float32)
                vals = tf.concat([vals, const], axis=1)

                indices = tf.concat([indices, tf.ones((beam_width, top_k - mod_top_k), dtype=tf.int32)], axis=1)
                self.ast_top_k_values.append(vals)
                self.ast_top_k_indices.append(indices)

        with tf.name_scope("optimization"):
            opt = tf.compat.v1.train.AdamOptimizer(config.learning_rate)
            self.gen_loss = self.ast_gen_loss
            regularizor = tf.reduce_sum([tf.square(sigma) for sigma in self.encoder.program_encoder.sigmas])
            self.loss = self.gen_loss #+ regularizor
            gen_train_ops = Model.get_var_list('both')
            self.train_op = opt.minimize(self.loss, var_list=gen_train_ops)



    def get_latent_state(self, sess, apis, types, kws,
                         return_type, formal_param_inputs,
                         fields, method, classname, javadoc_kws,
                         surr_ret, surr_fp, surr_method
                         ):
        feed = {self.apicalls: apis,
                self.types: types,
                self.keywords: kws,
                self.return_type: return_type,
                self.formal_param_inputs: formal_param_inputs,
                self.field_inputs: fields,
                self.method: method,
                self.classname: classname,
                self.javadoc_kws: javadoc_kws,
                self.surr_ret: surr_ret,
                self.surr_fp: surr_fp,
                self.surr_method: surr_method
                }

        [state, method_embedding] = sess.run([self.latent_state,
                                              self.encoder.program_encoder.surr_enc.internal_method_embedding], feed)
        return state, method_embedding


    # This method is called from infer.py, used for majority of experiments reported in the paper
    def get_initial_state(self, sess, apis, types, kws,
                          return_type, formal_param_inputs,
                          fields, method, classname, javadoc_kws,
                          surr_ret, surr_fp, surr_method,
                          visibility=1.00
                          ):

        feed = {self.apicalls: apis,
                self.types: types,
                self.keywords: kws,
                self.return_type: return_type,
                self.formal_param_inputs: formal_param_inputs,
                self.field_inputs: fields,
                self.method: method,
                self.classname: classname,
                self.javadoc_kws: javadoc_kws,
                self.surr_ret: surr_ret,
                self.surr_fp: surr_fp,
                self.surr_method: surr_method,
                self.encoder.ev_drop_rate: np.asarray([1.0 - visibility] * 10),
                }

        [state, method_embedding] = sess.run([self.initial_state,
                                              self.encoder.program_encoder.surr_enc.dropped_embedding], feed)
        return state, method_embedding

    def get_initial_state_from_latent_state(self, sess, latent_state):
        feed = {self.latent_state: latent_state}
        state = sess.run(self.initial_state, feed)
        return state

    def get_random_initial_state(self, sess):
        latent_state = np.random.normal(loc=0., scale=1.,
                                        size=(1, self.config.latent_size))
        latent_state = np.tile(latent_state, [self.config.batch_size, 1])
        initial_state = sess.run(self.initial_state,
                                 feed_dict={self.latent_state: latent_state})
        initial_state = np.transpose(np.array(initial_state), [1, 0, 2])  # batch-first
        return initial_state

    def get_initial_symtab(self, sess):
        init_symtab, init_unused_varflag, init_nullptr_varflag \
            = sess.run([self.decoder.program_decoder.ast_tree.init_symtab,
                        self.decoder.program_decoder.ast_tree.init_unused_varflag,
                        self.decoder.program_decoder.ast_tree.init_nullptr_varflag,
                        ])
        return [[x, y, z] for x, y, z in zip(init_symtab, init_unused_varflag, init_nullptr_varflag)]

    def get_next_ast_state(self, sess, ast_node, ast_edge, ast_state,
                           candies):

        feed = {self.nodes.name: np.array(ast_node, dtype=np.int32),
                self.edges.name: np.array(ast_edge, dtype=np.bool)}

        self.feed_inputs(feed, candies)

        for i in range(self.config.decoder.num_layers):
            feed[self.initial_state[i].name] = np.array(ast_state[i])

        feed[self.decoder.program_decoder.ast_tree.init_symtab] = np.array([candy.symtab for candy in candies])
        feed[self.decoder.program_decoder.ast_tree.init_unused_varflag] = np.array(
            [candy.init_unused_varflag for candy in candies])
        feed[self.decoder.program_decoder.ast_tree.init_nullptr_varflag] = np.array(
            [candy.init_nullptr_varflag for candy in candies])

        [ast_state, ast_symtab, unused_varflag, nullptr_varflag, beam_ids, beam_ln_probs] = sess.run(
            [self.decoder.program_decoder.ast_tree.state,
             self.decoder.program_decoder.ast_tree.symtab,
             self.decoder.program_decoder.ast_tree.unused_varflag,
             self.decoder.program_decoder.ast_tree.nullptr_varflag,
             self.ast_top_k_indices, self.ast_top_k_values], feed)

        return ast_state, ast_symtab, unused_varflag, nullptr_varflag, beam_ids, beam_ln_probs

    def feed_inputs(self, feed, candies):
        beam_width = len(candies)
        input_fp_depth = candies[0].formal_param_inputs.shape[0]
        input_field_depth = candies[0].field_types.shape[0]
        total_var_map_length = candies[0].mappers.shape[0]
        total_surr_methods = candies[0].method_embedding.shape[0]
        dimension = candies[0].method_embedding.shape[1]

        return_type = np.zeros(shape=(beam_width,), dtype=np.int32)
        formal_param_inputs = np.zeros(shape=(beam_width, input_fp_depth), dtype=np.int32)
        field_types = np.zeros(shape=(beam_width, input_field_depth), dtype=np.int32)

        var_decl_ids = np.zeros(shape=(beam_width, 1), dtype=np.int32)
        node_type_number = np.zeros(shape=(beam_width, 1), dtype=np.int32)

        all_var_mappers = np.zeros(shape=(beam_width, total_var_map_length), dtype=np.int32)
        type_helper_val = np.zeros(shape=(beam_width, 1), dtype=np.int32)
        expr_type_val = np.zeros(shape=(beam_width, 1), dtype=np.int32)
        ret_type_val = np.zeros(shape=(beam_width, 1), dtype=np.int32)

        ret_reached = np.zeros(shape=(beam_width, 1), dtype=np.bool)
        iattrib = np.zeros(shape=(beam_width, 1, 3), dtype=np.bool)

        method_embeddings = np.zeros(shape=(beam_width, total_surr_methods, dimension), dtype=np.float32)

        for batch_id, candy in enumerate(candies):
            return_type[batch_id] = candy.return_type
            formal_param_inputs[batch_id] = candy.formal_param_inputs
            field_types[batch_id] = candy.field_types

            var_decl_ids[batch_id] = [candy.var_decl_id]
            all_var_mappers[batch_id] = candy.mappers

            node_type_number[batch_id] = [candy.next_node_type]

            ret_type_val[batch_id], expr_type_val[batch_id], \
            type_helper_val[batch_id] = candy.get_ret_expr_helper_types()

            ret_reached[batch_id] = candy.get_return_reached()
            iattrib[batch_id] = candy.get_iattrib()

            method_embeddings[batch_id] = candy.method_embedding

        feed_extra = {
            self.var_decl_ids.name: np.array(var_decl_ids, dtype=np.int32),

            self.type_helper_val.name: np.array(type_helper_val, dtype=np.int32),
            self.expr_type_val.name: np.array(expr_type_val, dtype=np.int32),
            self.ret_type_val.name: np.array(ret_type_val, dtype=np.int32),

            self.ret_reached: ret_reached,
            self.iattrib: np.array(iattrib, dtype=np.bool),

            self.return_type: return_type,
            self.formal_param_inputs: formal_param_inputs,
            self.field_inputs: field_types,

            self.node_type_number: node_type_number,
            self.all_var_mappers: all_var_mappers,
            self.encoder.program_encoder.surr_enc.internal_method_embedding: method_embeddings,
        }

        feed.update(feed_extra)

        return

    def get_decoder_probs(self, sess, nodes, edges, targets, var_decl_ids, ret_reached, \
                          node_type_number, \
                          type_helper_val, expr_type_val, ret_type_val, \
                          all_var_mappers, iattrib, \
                          apis, types, kws,
                          return_type, formal_param_inputs,
                          fields, method, classname, javadoc_kws,
                          surr_ret, surr_fp, surr_method,
                          visibility=1.00
                          ):
        def calculate_mean_prob(matrix, targets, node_type_number, node_type):
            matrix_choices = np.argmax(matrix, axis=2)
            ast_probs = np.sum((matrix_choices == targets) * (node_type_number == node_type))
            num_nodes = np.sum(node_type_number == node_type)
            return ast_probs, num_nodes

        # feed the encoder
        feed = {self.apicalls: apis,
                self.types: types,
                self.keywords: kws,
                self.return_type: return_type,
                self.formal_param_inputs: formal_param_inputs,
                self.field_inputs: fields,
                self.method: method,
                self.classname: classname,
                self.javadoc_kws: javadoc_kws,
                self.surr_ret: surr_ret,
                self.surr_fp: surr_fp,
                self.surr_method: surr_method,
                self.encoder.ev_drop_rate: np.asarray([1.0 - visibility] * 10),
                }
        # feed the decoder
        feed.update({self.nodes: nodes, self.edges: edges, self.targets: targets})
        feed.update({self.var_decl_ids: var_decl_ids,
                     self.ret_reached: ret_reached,
                     self.iattrib: iattrib,
                     self.all_var_mappers: all_var_mappers})
        feed.update({self.node_type_number: node_type_number})
        feed.update({self.type_helper_val: type_helper_val, self.expr_type_val: expr_type_val,
                     self.ret_type_val: ret_type_val})

        concept_matrix, api_matrix, type_matrix, \
        clstype_matrix, var_matrix, op_matrix, \
        method_matrix, _ = sess.run(self.decoder.ast_logits, feed)



        concept_mean_prob = calculate_mean_prob(concept_matrix, targets, node_type_number, CONCEPT_NODE)
        api_mean_prob = calculate_mean_prob(api_matrix,  targets, node_type_number, API_NODE)
        type_mean_prob = calculate_mean_prob(type_matrix,  targets, node_type_number, TYPE_NODE)
        clstype_mean_prob = calculate_mean_prob(clstype_matrix,  targets, node_type_number, CLSTYPE_NODE)
        var_mean_prob = calculate_mean_prob(var_matrix,  targets, node_type_number, VAR_NODE)
        vardecl_mean_prob = calculate_mean_prob(var_matrix,  targets, node_type_number, VAR_DECL_NODE)
        op_mean_prob = calculate_mean_prob(op_matrix,  targets, node_type_number, OP_NODE)
        method_mean_prob = calculate_mean_prob(method_matrix,   targets, node_type_number, METHOD_NODE)

        return [concept_mean_prob, api_mean_prob, type_mean_prob,
                clstype_mean_prob, var_mean_prob, vardecl_mean_prob,
                op_mean_prob,
                method_mean_prob]


    @staticmethod
    def get_var_list(input):

        all_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        encoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        decoder_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')

        critical_vars, non_critical_vars = [], []
        critical_encoders = ['formal_param', 'field_vars', 'ret_type',
                             'method_kw', 'class_kw', 'javadoc_kw']
        non_critical_encoders = ['ast_tree_api', 'ast_tree_types', 'ast_tree_kw',
                                 'emb_apiname', 'emb_typename', 'emb_kw',
                                 'surrounding']
        for ev in critical_encoders:
            scp = 'encoder/Mean/' + ev
            critical_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scp)
        # critical_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder/Mean/sigma")

        # critical_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        #                                  scope='decoder/ast_tree/tree_decoder/method_prediction')

        for ev in non_critical_encoders:
            scp = 'encoder/Mean/' + ev
            non_critical_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scp)
        # non_critical_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder/Mean/sigma")

        var_dict = {'encoder': encoder_vars,
                    'decoder': decoder_vars,
                    'both': all_vars,
                    'critical_vars': critical_vars,
                    'non_critical_vars': non_critical_vars
                    }
        return var_dict[input]

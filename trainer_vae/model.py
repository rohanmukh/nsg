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
from collections import defaultdict, Counter, OrderedDict, namedtuple, deque
from typing import List, Dict, Any, Tuple, Iterable, Set, Optional

import numpy as np
import tensorflow as tf

from trainer_vae.architecture import Encoder, Decoder
from tensorflow.contrib import seq2seq
from synthesis.ops.candidate_ast import TYPE_NODE, VAR_NODE, API_NODE, SYMTAB_MOD, OP_NODE, CONCEPT_NODE, METHOD_NODE, \
    CLSTYPE_NODE, VAR_DECL_NODE
#from dpu_utils.tfmodels import AsyncGGNN
from gnn import AsyncGGNN
from trainer_vae.utils import construct_minibatch

#EXPANSION_LABELED_EDGE_TYPE_NAMES = ["Child"]
#EXPANSION_UNLABELED_EDGE_TYPE_NAMES = ["Parent", "NextUse", "NextToken", "NextSibling", "NextSubtree",
EXPANSION_LABELED_EDGE_TYPE_NAMES = []
EXPANSION_UNLABELED_EDGE_TYPE_NAMES = ["Child", "Parent", "NextSibling",
                                       "InheritedToSynthesised",
                                       "NextToken", "NextUse"]

class Model:
    def __init__(self, config, top_k=5, gnn_node_vocab=None):
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

        if self.config.decoder.ifnag:
            self.parameters = {}
            _ = self.get_decoder_default_hyperparameters()
            #self.hyperparameters = \
            #    self.decoder.program_decoder.ast_tree.hyperparameters

            self.gnn_node_vocab = gnn_node_vocab
            eg_token_vocab_size = len(gnn_node_vocab)
            eg_hidden_size = self.hyperparameters['eg_hidden_size']

            self.parameters['eg_token_embeddings'] = \
                tf.get_variable(name='eg_token_embeddings',
                                shape=[eg_token_vocab_size, eg_hidden_size],
                                initializer=tf.random_normal_initializer(),
                                )

            self.placeholders = {}
            eg_edge_type_num = len(EXPANSION_LABELED_EDGE_TYPE_NAMES) + len(
                EXPANSION_UNLABELED_EDGE_TYPE_NAMES)
            self.placeholders['eg_node_token_ids'] = tf.placeholder(
                tf.int32,
                [None],
                name="eg_node_token_ids")
            # Initial nodes I: Node IDs that will have no (active) incoming edges.
            self.placeholders['eg_initial_node_ids'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None],
                               name="eg_initial_node_ids")

            # Sending nodes S_{s,e}: Source node ids of edges of type e
            # propagating in step s. Restrictions: If v in S_{s,e}, then
            # v in R_{s'} for s' < s or v in I.
            self.placeholders['eg_sending_node_ids'] = \
                [[tf.placeholder(dtype=tf.int32,
                                shape=[None],
                                name="eg_sending_node_ids_step%i_edgetyp%i" % (
                                    step, edge_typ))
                for edge_typ in range(eg_edge_type_num)]
                for step in range(
                    self.hyperparameters['eg_propagation_substeps'])]

            # Normalised edge target nodes T_{s}: Targets of edges propagating
            # in step s, normalised to a continuous range starting from 0.
            # This is used for aggregating messages from the sending nodes.
            self.placeholders['eg_msg_target_node_ids'] = \
                [tf.placeholder(dtype=tf.int32,
                                shape=[None],
                                name="eg_msg_targets_nodes_step%i" % (step,))
                for step in range(
                    self.hyperparameters['eg_propagation_substeps'])]

            # Receiving nodes R_{s}: Target node ids of aggregated messages
            # in propagation step s. Restrictions: If v in R_{s}, v not in
            # R_{s'} for all s' != s and v not in I
            self.placeholders['eg_receiving_node_ids'] = \
                [tf.placeholder(dtype=tf.int32,
                                shape=[None],
                                name="eg_receiving_nodes_step%i" % (step,))
                for step in range(
                    self.hyperparameters['eg_propagation_substeps'])]

            # Number of receiving nodes N_{s}
            # Restrictions: N_{s} = len(R_{s})
            self.placeholders['eg_receiving_node_nums'] = \
                tf.placeholder(
                    dtype=tf.int32,
                    shape=[self.hyperparameters['eg_propagation_substeps']],
                    name="eg_receiving_nodes_nums")

            # We don't use context graph representation for now.
            eg_initial_node_representations = \
                tf.nn.embedding_lookup(
                    self.parameters['eg_token_embeddings'],
                    self.placeholders['eg_node_token_ids'])
            eg_hypers = {name.replace("eg_", "", 1): value
                        for (name, value) in self.hyperparameters.items()
                        if name.startswith("eg_")}
            eg_hypers['propagation_rounds'] = 1
            # We don't have labelled edge for now.
            eg_hypers['num_labeled_edge_types'] = len(
                EXPANSION_LABELED_EDGE_TYPE_NAMES)
            eg_hypers['num_unlabeled_edge_types'] = len(
                EXPANSION_UNLABELED_EDGE_TYPE_NAMES)

            with tf.variable_scope("ExpansionGraph"):
                eg_model = AsyncGGNN(eg_hypers)
                self.eg_node_representations = tf.identity(
                    eg_initial_node_representations)

                # Note that we only use a single async schedule here,
                # so every argument is wrapped in
                # [] to use the generic code supporting many schedules:
                self.eg_node_representations = \
                    eg_model.async_ggnn_layer(
                        eg_initial_node_representations,
                        [self.placeholders['eg_initial_node_ids']],
                        [self.placeholders['eg_sending_node_ids']],
                        [self.placeholders['eg_initial_node_ids']],
                        #[self.__embed_edge_labels(
                        #    self.hyperparameters['eg_propagation_substeps'])],
                        [self.placeholders['eg_msg_target_node_ids']],
                        [self.placeholders['eg_receiving_node_ids']],
                        [self.placeholders['eg_receiving_node_nums']])

                self.eg_node_representations = tf.reshape(
                    self.eg_node_representations,
                    [-1, self.config.max_ast_depth, eg_hidden_size])
                self.gnn_inputs = tf.unstack(tf.transpose(
                    self.eg_node_representations, [1, 0, 2]), axis=0)
                self.rnn_input = tf.transpose(self.eg_node_representations, [1, 0, 2])

        with tf.variable_scope("decoder"):
            latent_state_lifted = tf.layers.dense(self.latent_state, config.decoder.units)
            self.initial_state = [latent_state_lifted] * config.decoder.num_layers
            if self.config.decoder.ifnag:
                self.decoder = Decoder(
                    config, nodes, edges,
                    var_decl_ids, ret_reached, iattrib,
                    self.all_var_mappers,
                    type_helper_val, expr_type_val, ret_type_val,
                    node_type_number,
                    formal_param_inputs, field_inputs,
                    self.return_type,
                    self.encoder.program_encoder.surr_enc.internal_method_embedding,
                    self.initial_state,
                    gnn_inputs = self.gnn_inputs)
            else:
                self.decoder = Decoder(
                    config, nodes, edges,
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
                          visibility=1.00,
                          gnn_info=None
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

        gnn_minibatch, gnn_batch_data = construct_minibatch(gnn_info, self)
        feed.update(gnn_minibatch)

        concept_matrix, api_matrix, type_matrix, \
        clstype_matrix, var_matrix, vardecl_matrix, op_matrix, \
        method_matrix, _ = sess.run(self.decoder.ast_logits, feed)

        concept_mean_prob = calculate_mean_prob(concept_matrix, targets, node_type_number, CONCEPT_NODE)
        api_mean_prob = calculate_mean_prob(api_matrix,  targets, node_type_number, API_NODE)
        type_mean_prob = calculate_mean_prob(type_matrix,  targets, node_type_number, TYPE_NODE)
        clstype_mean_prob = calculate_mean_prob(clstype_matrix,  targets, node_type_number, CLSTYPE_NODE)
        var_mean_prob = calculate_mean_prob(var_matrix,  targets, node_type_number, VAR_NODE)
        vardecl_mean_prob = calculate_mean_prob(vardecl_matrix,  targets, node_type_number, VAR_DECL_NODE)
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

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        return {
                'optimizer': 'Adam',
                'seed': 0,
                'dropout_keep_rate': 0.9,
                'learning_rate': 0.00025,
                'learning_rate_decay': 0.98,
                'momentum': 0.85,
                'gradient_clip': 1,
                'max_epochs': 500,
                'patience': 5,
               }

    def get_decoder_default_hyperparameters(self) -> Dict[str, Any]:
        decoder_defaults = {
                    'eg_token_vocab_size': 100,
                    'eg_literal_vocab_size': 10,
                    'eg_max_variable_choices': 10,
                    #'eg_propagation_substeps': 100,
                    'eg_propagation_substeps': 85,
                    'eg_hidden_size': 64,
                    #'eg_edge_label_size': 16,
                    'eg_edge_label_size': 0,
                    'exclude_edge_types': [],

                    'eg_graph_rnn_cell': 'GRU',  # GRU or RNN
                    'eg_graph_rnn_activation': 'tanh',  # tanh, ReLU

                    'eg_use_edge_bias': False,

                    'eg_use_vars_for_production_choice': True,  # Use mean-pooled variable representation as input for production choice
                    'eg_update_last_variable_use_representation': True,

                    'eg_use_literal_copying': True,
                    'eg_use_context_attention': True,
                    'eg_max_context_tokens': 500,
                   }

        cg_defaults = self.get_default_hyperparameters_context_graph()
        defaults = self.get_default_hyperparameters()
        decoder_defaults.update(cg_defaults)
        decoder_defaults.update(defaults)
        self.hyperparameters = decoder_defaults
        return decoder_defaults

    def get_default_hyperparameters_context_graph(self) -> Dict[str, Any]:
        my_defaults = {
                        'max_num_cg_nodes_in_batch': 100000,

                        # Context Program Graph things:
                        'excluded_cg_edge_types': [],
                        'cg_add_subtoken_nodes': True,

                        'cg_node_label_embedding_style': 'Token',  # One of ['Token', 'CharCNN']
                        'cg_node_label_vocab_size': 10000,
                        'cg_node_label_char_length': 16,
                        "cg_node_label_embedding_size": 32,

                        'cg_node_type_vocab_size': 5000,
                        'cg_node_type_max_num': 10,
                        'cg_node_type_embedding_size': 32,

                        "cg_ggnn_layer_timesteps": [3, 1, 3, 1],
                        "cg_ggnn_residual_connections": {"1": [0], "3": [0, 1]},

                        "cg_ggnn_hidden_size": 64,
                        "cg_ggnn_use_edge_bias": False,
                        "cg_ggnn_use_edge_msg_avg_aggregation": False,
                        "cg_ggnn_use_propagation_attention": False,
                        "cg_ggnn_graph_rnn_activation": "tanh",
                        "cg_ggnn_graph_rnn_cell": "GRU",

                     }
        return my_defaults


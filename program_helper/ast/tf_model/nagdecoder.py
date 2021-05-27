from collections import defaultdict, Counter, OrderedDict, namedtuple, deque
from typing import List, Dict, Any, Tuple, Iterable, Set, Optional

import tensorflow as tf

from program_helper.ast.tf_model.base_ast_encoder import BaseTreeEncoding
from utilities.tensor_permutor import permute_batched_tensor_3dim, permute_batched_tensor_2dim
from dpu_utils.tfmodels import AsyncGGNN
from synthesis.ops.candidate_ast import SYMTAB_MOD, VAR_NODE, TYPE_NODE, API_NODE, OP_NODE, METHOD_NODE, CLSTYPE_NODE, \
    VAR_DECL_NODE

BIG_NUMBER = 1e7
SMALL_NUMBER = 1e-7

# These are special non-terminal symbols that are expanded to literals, either from
# a small dict that we collect during training, or by copying from the context.
ROOT_NONTERMINAL = 'Expression'
VARIABLE_NONTERMINAL = 'Variable'
LITERAL_NONTERMINALS = ['IntLiteral', 'CharLiteral', 'StringLiteral']
LAST_USED_TOKEN_NAME = '<LAST TOK>'
#EXPANSION_LABELED_EDGE_TYPE_NAMES = ["Child"]
#EXPANSION_UNLABELED_EDGE_TYPE_NAMES = ["Parent", "NextUse", "NextToken", "NextSibling", "NextSubtree",
#                                       "InheritedToSynthesised"]
EXPANSION_LABELED_EDGE_TYPE_NAMES = []
EXPANSION_UNLABELED_EDGE_TYPE_NAMES = ["Child", "Parent", "NextSibling",
                                       "InheritedToSynthesised",
                                       "NextToken", "NextUse"]


class NAGDecoder(BaseTreeEncoding):
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
                 gnn_inputs=None
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
                var_output_logits, vardecl_output_logits, concept_output_logits,\
                op_output_logits, method_output_logits = [], [], [], [], [], [], [], []
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
                                                   internal_var_mapper,
                                                   gnn_inputs[i]
                                           )

                api_logit, type_logit, clstype_logit, var_logit, vardecl_logit, \
                        concept_logit, op_logit, method_logit = logits

                api_output_logits.append(api_logit)
                type_output_logits.append(type_logit)
                clstype_output_logits.append(clstype_logit)
                var_output_logits.append(var_logit)
                vardecl_output_logits.append(vardecl_logit)
                concept_output_logits.append(concept_logit)
                op_output_logits.append(op_logit)
                method_output_logits.append(method_logit)

        self.output_logits = [
            tf.stack(concept_output_logits, 1),
            tf.stack(api_output_logits, 1),
            tf.stack(type_output_logits, 1),
            tf.stack(clstype_output_logits, 1),
            tf.stack(var_output_logits, 1),
            tf.stack(vardecl_output_logits, 1),
            tf.stack(op_output_logits, 1),
            tf.stack(method_output_logits, 1),
            tf.ones((batch_size, len(nodes), batch_size), dtype=tf.float32)
        ]

        #self.make_parameters()

    # ---------- Constructing the core model ----------
    def make_parameters(self):
        self.parameters = {}
        label_embedding_size = self.hyperparameters['cg_node_label_embedding_size']
        if self.hyperparameters['cg_node_label_embedding_style'].lower() == 'token':
            type_vocab_size = self.hyperparameters['cg_node_label_vocab_size']
            self.parameters['cg_node_label_embeddings'] = \
                tf.get_variable(name='cg_node_label_embeddings',
                                shape=[type_vocab_size, label_embedding_size],
                                initializer=tf.random_normal_initializer())

        type_embedding_size = self.hyperparameters['cg_node_type_embedding_size']
        if type_embedding_size > 0:
            type_vocab_size = self.hyperparameters['cg_node_type_vocab_size']
            self.parameters['cg_node_type_embeddings'] = \
                tf.get_variable(name='cg_node_type_embeddings',
                                shape=[type_vocab_size, type_embedding_size],
                                initializer=tf.random_normal_initializer())

        # Use an OrderedDict so that we can rely on the iteration order later.
        self.__expansion_labeled_edge_types, self.__expansion_unlabeled_edge_types = \
            get_restricted_edge_types(self.hyperparameters)

        #eg_token_vocab_size = len(self.metadata['eg_token_vocab'])
        eg_token_vocab_size = self.hyperparameters['eg_token_vocab_size']
        eg_hidden_size = self.hyperparameters['eg_hidden_size']
        eg_edge_label_size = self.hyperparameters['eg_edge_label_size']
        self.parameters['eg_token_embeddings'] = \
            tf.get_variable(name='eg_token_embeddings',
                            shape=[eg_token_vocab_size, eg_hidden_size],
                            initializer=tf.random_normal_initializer(),
                            )

        # TODO: Should be more generic than being fixed to the number productions...
        if eg_edge_label_size > 0:
            self.parameters['eg_edge_label_embeddings'] = \
                tf.get_variable(name='eg_edge_label_embeddings',
                                shape=[self.hyperparameters['eg_edge_label_vocab_size'], eg_edge_label_size],
                                initializer=tf.random_normal_initializer(),
                                )

    def make_placeholders(self):
        self.placeholders['eg_node_token_ids'] = \
            tf.placeholder(tf.int32,
                           [None],
                           name="eg_node_token_ids")

        # List of lists of lists of (embeddings of) labels of edges L_{r,s,e}:
        # Labels of edges of type
        # e propagating in step s.
        # Restrictions: len(L_{s,e}) = len(S_{s,e})  [see __make_train_placeholders]
        self.placeholders['eg_edge_label_ids'] = \
            [[tf.placeholder(dtype=tf.int32,
                             shape=[None],
                             name="eg_edge_label_step%i_typ%i" % (step, edge_typ))
              for edge_typ in range(len(self.__expansion_labeled_edge_types))]
             for step in range(self.hyperparameters['eg_propagation_substeps'])]

        if is_train:
            self.__make_train_placeholders()
        else:
            self.__make_test_placeholders()

    def __make_train_placeholders(self):
        eg_edge_type_num = len(EXPANSION_LABELED_EDGE_TYPE_NAMES) + len(
            EXPANSION_UNLABELED_EDGE_TYPE_NAMES)
        # Initial nodes I: Node IDs that will have no (active) incoming edges.
        self.placeholders['eg_initial_node_ids'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name="eg_initial_node_ids")

        # Sending nodes S_{s,e}: Source node ids of edges of type e propagating in step s.
        # Restrictions: If v in S_{s,e}, then v in R_{s'} for s' < s or v in I.
        self.placeholders['eg_sending_node_ids'] = \
            [[tf.placeholder(dtype=tf.int32,
                             shape=[None],
                             name="eg_sending_node_ids_step%i_edgetyp%i" % (step, edge_typ))
              for edge_typ in range(eg_edge_type_num)]
             for step in range(self.hyperparameters['eg_propagation_substeps'])]

        # Normalised edge target nodes T_{s}: Targets of edges propagating in step s, normalised to a
        # continuous range starting from 0. This is used for aggregating messages from the sending nodes.
        self.placeholders['eg_msg_target_node_ids'] = \
            [tf.placeholder(dtype=tf.int32,
                            shape=[None],
                            name="eg_msg_targets_nodes_step%i" % (step,))
             for step in range(self.hyperparameters['eg_propagation_substeps'])]

        # Receiving nodes R_{s}: Target node ids of aggregated messages in
        # propagation step s.
        # Restrictions: If v in R_{s}, v not in R_{s'} for all s' != s and v
        # not in I
        self.placeholders['eg_receiving_node_ids'] = \
            [tf.placeholder(dtype=tf.int32,
                            shape=[None],
                            name="eg_receiving_nodes_step%i" % (step,))
             for step in range(self.hyperparameters['eg_propagation_substeps'])]

        # Number of receiving nodes N_{s}
        # Restrictions: N_{s} = len(R_{s})
        self.placeholders['eg_receiving_node_nums'] = \
            tf.placeholder(dtype=tf.int32,
                           shape=[self.hyperparameters['eg_propagation_substeps']],
                           name="eg_receiving_nodes_nums")

    def __make_test_placeholders(self):
        eg_h_dim = self.hyperparameters['eg_hidden_size']
        self.placeholders['eg_production_node_representation'] = \
            tf.placeholder(dtype=tf.float32,
                           shape=[eg_h_dim],
                           name="eg_production_node_representation")

        if self.hyperparameters['eg_use_vars_for_production_choice']:
            self.placeholders['eg_production_var_representations'] = \
                tf.placeholder(dtype=tf.float32,
                               shape=[None, eg_h_dim],
                               name="eg_production_var_representations")

        if self.hyperparameters["eg_use_literal_copying"] or self.hyperparameters['eg_use_context_attention']:
            self.placeholders['context_token_representations'] = \
                tf.placeholder(dtype=tf.float32,
                               shape=[self.hyperparameters['eg_max_context_tokens'], eg_h_dim],
                               name='context_token_representations')

        self.placeholders['eg_varproduction_node_representation'] = \
            tf.placeholder(dtype=tf.float32,
                           shape=[eg_h_dim],
                           name="eg_varproduction_node_representation")
        self.placeholders['eg_num_variable_choices'] = \
            tf.placeholder(dtype=tf.int32, shape=[], name='eg_num_variable_choices')
        self.placeholders['eg_varproduction_options_representations'] = \
            tf.placeholder(dtype=tf.float32,
                           shape=[None, eg_h_dim],
                           name="eg_varproduction_options_representations")

        self.placeholders['eg_litproduction_node_representation'] = \
            tf.placeholder(dtype=tf.float32,
                           shape=[eg_h_dim],
                           name="eg_litproduction_node_representation")
        if self.hyperparameters['eg_use_literal_copying']:
            self.placeholders['eg_litproduction_choice_normalizer'] = \
                tf.placeholder(dtype=tf.int32,
                               shape=[None],
                               name="eg_litproduction_choice_normalizer")

        # Used for one-step message propagation of expansion graph:
        eg_edge_type_num = len(EXPANSION_LABELED_EDGE_TYPE_NAMES) + len(
            EXPANSION_UNLABELED_EDGE_TYPE_NAMES)
        self.placeholders['eg_msg_source_representations'] = \
            [tf.placeholder(dtype=tf.float32,
                            shape=[None, eg_h_dim],
                            name="eg_msg_source_representations_etyp%i" % (edge_typ,))
             for edge_typ in range(eg_edge_type_num)]

        self.placeholders['eg_msg_target_label_id'] =\
            tf.placeholder(dtype=tf.int32, shape=[], name='eg_msg_target_label_id')

    def __make_train_model(self):
        # We don't use context graph representation for now.
        eg_initial_node_representations = \
            tf.nn.embedding_lookup(self.parameters['eg_token_embeddings'],
                                   self.placeholders['eg_node_token_ids'])
        eg_hypers = {name.replace("eg_", "", 1): value
                     for (name, value) in self.hyperparameters.items()
                     if name.startswith("eg_")}
        # We don't have labelled edge for now.
        eg_hypers['num_labeled_edge_types'] = len(
            EXPANSION_LABELED_EDGE_TYPE_NAMES)
        eg_hypers['num_unlabeled_edge_types'] = len(
            EXPANSION_UNLABELED_EDGE_TYPE_NAMES)
        with tf.variable_scope("ExpansionGraph"):
            eg_model = AsyncGGNN(eg_hypers)

            # Note that we only use a single async schedule here, so every argument is wrapped in
            # [] to use the generic code supporting many schedules:
            eg_node_representations = \
                eg_model.async_ggnn_layer(
                    eg_initial_node_representations,
                    [self.placeholders['eg_initial_node_ids']],
                    [self.placeholders['eg_sending_node_ids']],
                    [self.__embed_edge_labels(
                        self.hyperparameters['eg_propagation_substeps'])],
                    [self.placeholders['eg_msg_target_node_ids']],
                    [self.placeholders['eg_receiving_node_ids']],
                    [self.placeholders['eg_receiving_node_nums']])

    def get_next_output(self, node, edge,
                        var_decl_id, ret_reached, iattrib,
                        type_helper_val, expr_type_val, ret_type_val,
                        node_type_number,
                        symtab_in, unused_varflag_in, nullptr_varflag_in,
                        method_ret_type_helper, method_fp_type_emb,
                        method_field_type_emb, internal_method_embedding,
                        state_in, internal_var_mapper, gnn_node):

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
            #input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
            #                   flat_varflag,
            #                   flat_nullptr_varflag_in,
            #                   ret_reached,
            #                   iattrib,
            #                   method_ret_type_helper,
            #                   flat_symtab,
            #                   ], axis=1)
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               ], axis=1)
            concept_output, concept_state = self.concept_encoder.get_next_output_with_symtab(input, edge,
                                                                                             state_in)
            concept_logit = self.concept_encoder.get_projection(concept_output)

        with tf.variable_scope('api_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               #flat_varflag,
                               #flat_nullptr_varflag_in,
                               #ret_reached,
                               #method_ret_type_helper,
                               #iattrib
                               ], axis=1)
            api_output, api_state = self.api_encoder.get_next_output_with_symtab(input, state_in)
            api_logit = self.api_encoder.get_projection(api_output)

        with tf.variable_scope('type_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               #flat_varflag,
                               #flat_nullptr_varflag_in,
                               #ret_reached,
                               #method_ret_type_helper,
                               #iattrib
                               ], axis=1)
            type_output, type_state = self.type_encoder.get_next_output_with_symtab(input, state_in)
            type_logit = self.type_encoder.get_projection(type_output)

        with tf.variable_scope('clstype_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               #flat_varflag,
                               #flat_nullptr_varflag_in,
                               #ret_reached,
                               #method_ret_type_helper,
                               #iattrib
                               ], axis=1)
            clstype_output, clstype_state = self.clstype_encoder.get_next_output_with_symtab(input, state_in)
            clstype_logit = self.clstype_encoder.get_projection(clstype_output)

        with tf.variable_scope('op_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               #flat_varflag,
                               #flat_nullptr_varflag_in,
                               #ret_reached,
                               #iattrib
                               ], axis=1)
            op_output, op_state = self.op_encoder.get_next_output_with_symtab(input, state_in)
            op_logit = self.op_encoder.get_projection(op_output)

        with tf.variable_scope('method_prediction'):
            internal_method_embedding_flat = tf.reshape(internal_method_embedding, (self.batch_size, -1))
            #input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
            #                   internal_method_embedding_flat,
            #                   flat_varflag,
            #                   flat_nullptr_varflag_in,
            #                   ret_reached,
            #                   tf.nn.embedding_lookup(self.type_emb, type_helper_val),
            #                   gnn_node
            #                ], axis=1)
            #input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
            #                   tf.nn.embedding_lookup(self.type_emb, type_helper_val),
            #                   gnn_node
            #                   ], axis=1)
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               ], axis=1)
            method_output, method_state = self.method_encoder.get_next_output_with_symtab(input, state_in)
            method_logit = self.method_encoder.get_projection(method_output)

        with tf.variable_scope('var_access_prediction_type'):
            #input1 = tf.concat([flat_symtab,
            #                    flat_varflag,
            #                    flat_nullptr_varflag_in,
            #                    tf.nn.embedding_lookup(self.type_emb, type_helper_val),
            #                    ], axis=1)
            #input1 = tf.concat([
            #                   tf.nn.embedding_lookup(self.type_emb, type_helper_val),
            #                   gnn_node
            #                   ], axis=1)
            input1 = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               ], axis=1)
            input1 = tf.layers.dense(input1, self.units, activation=tf.nn.tanh)
            input1 = tf.layers.dense(input1, self.units, activation=tf.nn.tanh)
            var_output1, var_state1 = self.var_encoder1.get_next_output_with_symtab(input1, state_in)

        with tf.variable_scope('var_access_prediction_exp'):
            #input2 = tf.concat([flat_symtab,
            #                    flat_varflag,
            #                    flat_nullptr_varflag_in,
            #                    tf.nn.embedding_lookup(self.type_emb, expr_type_val),
            #                    ], axis=1)
            input2 = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               ], axis=1)
            #input2 = tf.concat([tf.nn.embedding_lookup(self.type_emb, expr_type_val),
            #                   gnn_node
            #                   ], axis=1)
            input2 = tf.layers.dense(input2, self.units, activation=tf.nn.tanh)
            input2 = tf.layers.dense(input2, self.units, activation=tf.nn.tanh)
            var_output2, var_state2 = self.var_encoder2.get_next_output_with_symtab(input2, state_in)

        with tf.variable_scope('var_access_prediction_ret'):
            #input3 = tf.concat([flat_symtab,
            #                    flat_varflag,
            #                    flat_nullptr_varflag_in,
            #                    tf.nn.embedding_lookup(self.type_emb, ret_type_val),
            #                    ], axis=1)
            input3 = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               ], axis=1)
            #input3 = tf.concat([tf.nn.embedding_lookup(self.type_emb, ret_type_val),
            #                   gnn_node
            #                   ], axis=1)
            input3 = tf.layers.dense(input3, self.units, activation=tf.nn.tanh)
            input3 = tf.layers.dense(input3, self.units, activation=tf.nn.tanh)
            var_output3, var_state3 = self.var_encoder3.get_next_output_with_symtab(input3, state_in)

        with tf.variable_scope('var_declaration'):
            #input4 = tf.concat([flat_symtab,
            #                    flat_varflag,
            #                    flat_nullptr_varflag_in,
            #                    ], axis=1)
            #input4 = tf.concat([flat_symtab,
            #                    flat_varflag,
            #                    flat_nullptr_varflag_in
            #                    ], axis=1)
            input4 = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               gnn_node
                               ], axis=1)
            input4 = tf.layers.dense(input4, self.units, activation=tf.nn.tanh)
            input4 = tf.layers.dense(input4, self.units, activation=tf.nn.tanh)
            vardecl_output, vardecl_state = self.var_encoder4.get_next_output_with_symtab(input4, state_in)
            vardecl_logit = self.var_encoder4.get_projection(vardecl_output)

        # or nots
        symtabmod_or_not = tf.equal(node_type_number, SYMTAB_MOD)
        var_or_not = tf.equal(node_type_number, VAR_NODE)
        vardecl_or_not = tf.equal(node_type_number, VAR_DECL_NODE)
        type_or_not = tf.equal(node_type_number, TYPE_NODE)
        clstype_or_not = tf.equal(node_type_number, CLSTYPE_NODE)
        api_or_not = tf.equal(node_type_number, API_NODE)
        op_or_not = tf.equal(node_type_number, OP_NODE)
        method_or_not = tf.equal(node_type_number, METHOD_NODE)


        var_state = [tf.where(tf.not_equal(type_helper_val, 0),
                              var_state1[j],
                              tf.where(tf.not_equal(expr_type_val, 0),
                                       var_state2[j],
                                       tf.where(tf.not_equal(ret_type_val, 0),
                                                var_state3[j],
                                                var_state3[j]
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
                                               self.var_encoder3.get_projection(var_output3)
                                               )
                                      )
                             )


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
                                                                      tf.where(vardecl_or_not,
                                                                               vardecl_state[j],
                                                                               concept_state[j]
                                                                               )
                                                                      )
                                                            )
                                                    )
                                           )
                                       )
                              )
                          )
                 for j in range(self.num_layers)]

        logits = [api_logit, type_logit, clstype_logit,
                  var_logit, vardecl_logit,  concept_logit, op_logit,
                  method_logit]

        return state, stripped_symtab, stripped_new_unused_vars, stripped_new_nullptr_varflag, logits


def get_restricted_edge_types(hyperparameters: Dict[str, Any]):
    expansion_labeled_edge_types = OrderedDict(
        (name, edge_id) for (edge_id, name) in enumerate(
            n for n in EXPANSION_LABELED_EDGE_TYPE_NAMES))

    expansion_unlabeled_edge_types = OrderedDict(
        (name, edge_id) for (edge_id, name) in enumerate(
            n for n in EXPANSION_UNLABELED_EDGE_TYPE_NAMES))

    return expansion_labeled_edge_types, expansion_unlabeled_edge_types



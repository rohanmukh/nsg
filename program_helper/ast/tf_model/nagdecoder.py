from collections import defaultdict, Counter, OrderedDict, namedtuple, deque
from typing import List, Dict, Any, Tuple, Iterable, Set, Optional

import tensorflow as tf

from program_helper.ast.tf_model.base_ast_encoder import BaseTreeEncoding
from utilities.tensor_permutor import permute_batched_tensor_3dim
from dpu_utils.tfmodels import AsyncGGNN

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
                                       "InheritedToSynthesised"]


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
                                                   internal_var_mapper
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

        _ = self.get_decoder_default_hyperparameters()
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

        # Receiving nodes R_{s}: Target node ids of aggregated messages in propagation step s.
        # Restrictions: If v in R_{s}, v not in R_{s'} for all s' != s and v not in I
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

    def get_decoder_default_hyperparameters(self) -> Dict[str, Any]:
        decoder_defaults = {
                    'eg_token_vocab_size': 100,
                    'eg_literal_vocab_size': 10,
                    'eg_max_variable_choices': 10,
                    'eg_propagation_substeps': 100,
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

def get_restricted_edge_types(hyperparameters: Dict[str, Any]):
    expansion_labeled_edge_types = OrderedDict(
        (name, edge_id) for (edge_id, name) in enumerate(
            n for n in EXPANSION_LABELED_EDGE_TYPE_NAMES))

    expansion_unlabeled_edge_types = OrderedDict(
        (name, edge_id) for (edge_id, name) in enumerate(
            n for n in EXPANSION_UNLABELED_EDGE_TYPE_NAMES))

    return expansion_labeled_edge_types, expansion_unlabeled_edge_types

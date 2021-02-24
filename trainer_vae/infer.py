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

from __future__ import print_function

import random

import tensorflow as tf
import os
import json
import numpy as np

from trainer_vae.model import Model
from trainer_vae.utils import read_config


class BayesianPredictor(object):

    def __init__(self, save_dir, depth=None, batch_size=None, seed=None):

        if seed is not None:
            tf.compat.v1.set_random_seed(seed)
            np.random.seed(seed=seed)
            random.seed(seed)

        config_file = os.path.join(save_dir, 'config.json')
        with open(config_file) as f:
            self.config = read_config(json.load(f), infer=True)

        if depth is not None:
            self.config.max_ast_depth = 1

        if batch_size is not None:
            self.config.batch_size = batch_size

        self.config.trunct_num_batch = None

        self.model = Model(self.config, top_k=batch_size)

        self.sess = tf.Session()
        self.restore(save_dir)

    def restore(self, save):
        # restore the saved model
        vars_ = Model.get_var_list('both')
        old_saver = tf.compat.v1.train.Saver(vars_)
        ckpt = tf.train.get_checkpoint_state(save)
        old_saver.restore(self.sess, ckpt.model_checkpoint_path)

        return

    def close(self):
        self.sess.close()
        tf.reset_default_graph()
        return

    def get_latent_state(self, apis, types, kws,
                         return_type, formal_param_inputs,
                         fields, method, classname, javadoc_kws,
                         surr_ret, surr_fp, surr_method
                         ):
        state, method_embedding = self.model.get_latent_state(self.sess, apis, types, kws,
                                                              return_type, formal_param_inputs,
                                                              fields, method, classname, javadoc_kws,
                                                              surr_ret, surr_fp, surr_method
                                                              )
        return state, method_embedding


    # Get initial state is used for majority of the test cases in the paper, like semantic check and relevance check
    def get_initial_state(self, apis, types, kws,
                          return_type, formal_param_inputs,
                          fields, method, classname, javadoc_kws,
                          surr_ret, surr_fp, surr_method,
                          visibility=1.00
                          ):
        state, method_embedding = self.model.get_initial_state(self.sess, apis, types, kws,
                                                               return_type, formal_param_inputs,
                                                               fields, method, classname, javadoc_kws,
                                                               surr_ret, surr_fp, surr_method,
                                                               visibility=visibility
                                                               )
        return state, method_embedding

    def get_initial_state_from_latent_state(self, latent_state):
        init_state = self.model.get_initial_state_from_latent_state(self.sess, latent_state)
        return init_state

    def get_random_initial_state(self):
        state = self.model.get_random_initial_state(self.sess)
        return state

    def get_initial_symtab(self):
        symtab = self.model.get_initial_symtab(self.sess)
        return symtab

    def get_next_ast_state(self, ast_node, ast_edge, ast_state,
                           candies):
        ast_state, ast_symtab, unused_varflag, nullptr_varflag, beam_ids, beam_ln_probs = \
            self.model.get_next_ast_state(self.sess, ast_node, ast_edge,
                                          ast_state,
                                          candies)

        return ast_state, ast_symtab, unused_varflag, nullptr_varflag, beam_ids, beam_ln_probs

    def get_initial_state_from_next_batch(self, loader_batch, visibility=1.00):
        nodes, edges, targets, var_decl_ids, ret_reached, \
        node_type_number, \
        type_helper_val, expr_type_val, ret_type_val, \
        all_var_mappers, iattrib, \
        ret_type, fp_in, fields, \
        apis, types, kws, method, classname, javadoc_kws, \
        surr_ret, surr_fp, surr_method = loader_batch

        psi, method_embedding = self.get_initial_state(apis, types, kws,
                                                       ret_type, fp_in, fields, method, classname, javadoc_kws,
                                                       surr_ret, surr_fp, surr_method, visibility
                                                       )

        return psi, all_var_mappers, method_embedding

    def get_api_prob_from_next_batch(self, loader_batch, visibility=1.00):
        nodes, edges, targets, var_decl_ids, ret_reached, \
        node_type_number, \
        type_helper_val, expr_type_val, ret_type_val, \
        all_var_mappers, iattrib, \
        ret_type, fp_in, fields, \
        apis, types, kws, method, classname, javadoc_kws, \
        surr_ret, surr_fp, surr_method = loader_batch

        [concept_prob, api_prob, type_prob, clstype_prob, var_prob, vardecl_prob, op_prob, method_prob] \
                                    = self.model.get_decoder_probs(self.sess,
                                                nodes, edges, targets, var_decl_ids, ret_reached, \
                                                node_type_number, \
                                                type_helper_val, expr_type_val, ret_type_val, \
                                                all_var_mappers, iattrib, \
                                                apis, types, kws,
                                                ret_type, fp_in, fields, method, classname, javadoc_kws,
                                                surr_ret, surr_fp, surr_method, visibility=visibility
                                                )

        return concept_prob, api_prob, type_prob, clstype_prob, var_prob, vardecl_prob, op_prob, method_prob

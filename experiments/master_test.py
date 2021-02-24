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
import argparse
import json
import sys
import os

import numpy as np

from data_extraction.data_reader.data_loader import Loader
from program_helper.ast.parser.ast_exceptions import UndeclaredVarException, TypeMismatchException, VoidProgramException
from program_helper.ast.parser.ast_gen_checker import AstGenChecker
from program_helper.ast.visualize.ast_visualizor import AstVisualizer
from program_helper.program_reverse_map import ProgramRevMapper
from synthesis.json_synthesis import JSON_Synthesis
from synthesis.ops.candidate_ast import API_NODE
from synthesis.write_java import Write_Java
from experiments.jaccard_metric.get_jaccard_metrics import helper
from experiments.jaccard_metric.utils import plotter
from experiments.tSNE_visualizor import get_api
from experiments.tSNE_visualizor import fitTSNEandplot
from trainer_vae.infer import BayesianPredictor
from program_helper.program_beam_searcher import ProgramBeamSearcher
from utilities.basics import conditional_director_creator, dump_file, dump_json, dump_java
from utilities.logging import create_logger
from utilities.vocab_building_dictionary import DELIM
from utils import read_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


class MasterTester:
    def __init__(self, model_path=None,
                 save_path=None,
                 data_path=None,
                 beam_width=10,
                 seed=501):

        config_file = os.path.join(model_path, 'config.json')
        with open(config_file) as f:
            self.config = read_config(json.load(f), infer=True)

        self.encoder = BayesianPredictor(model_path, batch_size=beam_width, seed=seed, type='encoder')
        self.prog_mapper = ProgramRevMapper(self.encoder.config.vocab)
        self.decoder = BayesianPredictor(model_path, batch_size=beam_width, depth='change', seed=seed, type='decoder')
        self.program_beam_searcher = ProgramBeamSearcher(self.decoder)
        logger = create_logger(os.path.join(clargs.data, 'data_read.log'))

        self.ast_checker = AstGenChecker(self.program_beam_searcher.infer_model.config.vocab, logger=logger)
        self.ast_visualizer = AstVisualizer()
        self.json_synthesis, self.jsons_synthesized = JSON_Synthesis(), list()
        self.java_synthesis, self.javas_synthesized = Write_Java(), list()

        self.config.batch_size = beam_width
        self.loader = Loader(data_path, self.config)

        self.states, self.labels = [], []
        self.ast_nodes, self.fp_nodes, self.ret_types, self.fp_paths, self.ast_paths = [], [], [], [], []

        self.saver_dir = save_path

    def clear(self):
        self.ast_nodes.clear()
        self.fp_nodes.clear()
        self.ret_types.clear()
        self.fp_paths.clear()
        self.ast_paths.clear()
        self.jsons_synthesized.clear()
        self.javas_synthesized.clear()

    def close(self):
        self.encoder.close()
        self.decoder.close()

    def get_states(self, num_batches=2000):

        if self.config.trunct_num_batch is None:
            num_batches = num_batches
        else:
            num_batches = min(self.config.trunct_num_batch, num_batches)
        assert self.config.num_batches > 0, 'Not enough data'

        for i in range(num_batches):
            nodes, edges, targets, \
            var_decl_ids, \
            node_type_numbers, \
            type_helper_val, expr_type_val, ret_type_val, \
            ret_type, \
            fp_nodes, fp_edges, fp_type_targets, fp_type_or_not, fp_in, \
            apis, types, kws = self.loader.next_batch()
            state = self.encoder.get_latent_state(apis, types, kws,
                                                  ret_type, fp_in,
                                                  targets, edges, var_decl_ids,
                                                  node_type_numbers,
                                                  type_helper_val, expr_type_val, ret_type_val
                                                  )
            self.states.extend(state)

            api_or_not = node_type_numbers == API_NODE
            for t, api_bool in zip(targets, api_or_not):
                label = get_api(self.encoder.config, t, api_bool)
                self.labels.append(label)
            self.prog_mapper.add_data(nodes, edges, targets,
                                      var_decl_ids,
                                      node_type_numbers,
                                      type_helper_val, expr_type_val, ret_type_val,
                                      fp_nodes, fp_edges, fp_type_targets, fp_type_or_not, fp_in,
                                      ret_type,
                                      apis, types, kws)

    def plot_tsne(self, filename=None):
        new_states, new_labels = [], []
        for state, label in zip(self.states, self.labels):
            if label != 'N/A':
                new_states.append(state)
                new_labels.append(label)
        print('Fitting tSNE')
        path = os.path.join(self.saver_dir, filename)
        conditional_director_creator(path)
        path = os.path.join(path, filename)
        fitTSNEandplot(new_states, new_labels, path)

    def plot_jaccard(self, num_centroids=10, filename=None):
        new_states, new_labels = [], []
        for state, label in zip(self.states, self.labels):
            if len(label) != 0:
                new_states.append(state)
                new_labels.append(label)

        print('API Call Jaccard Calculations')
        jac_api_matrix, jac_api_vector = helper(new_states, new_labels, num_centroids=num_centroids)
        path = os.path.join(self.saver_dir, filename)
        conditional_director_creator(path)
        path = os.path.join(path, filename)
        plotter(jac_api_matrix, jac_api_vector, name=path)

    def get_memory_programs(self, num_data=100, filename='memory'):
        print("Doing a memory test")
        self.clear()
        self.ast_nodes.clear(),
        self.fp_nodes.clear()
        self.ret_types.clear()
        self.fp_paths.clear()
        self.ast_paths.clear()
        for i in range(num_data):
            temp_ = self.decoder.get_initial_state_from_latent_state(
                [self.states[i] for _ in range(self.decoder.config.batch_size)])
            psi_ = np.transpose(np.array(temp_), [1, 0, 2])  # batch_first
            # temp = [temp_ for _ in range(self.decoder.config.batch_size)]
            ast_nodes, fp_nodes, ret_types, ast_paths, fp_paths = \
                self.program_beam_searcher.beam_search_memory(initial_state=psi_,
                                                              ret_type=self.prog_mapper.get_return_type(i),
                                                              fp_types=self.prog_mapper.get_fp_type_inputs(i)
                                                              )
            self.ast_nodes.append(ast_nodes)
            self.fp_nodes.append(fp_nodes)
            self.ret_types.append(ret_types)
            self.fp_paths.append(fp_paths)
            self.ast_paths.append(ast_paths)
        path = os.path.join(self.saver_dir, filename)
        conditional_director_creator(path)
        self.synthesize(debug_print=False,
                        prog_mapper_print=True,
                        saver_path=path)

    def get_random_programs(self, num_data=100, filename='random'):
        print("Doing a random test")
        self.clear()
        for i in range(num_data):
            ast_nodes, fp_nodes, ret_types, ast_paths, fp_paths = self.program_beam_searcher.beam_search_random()
            self.ast_nodes.append(ast_nodes)
            self.fp_nodes.append(fp_nodes)
            self.ret_types.append(ret_types)
            self.fp_paths.append(fp_paths)
            self.ast_paths.append(ast_paths)
        path = os.path.join(self.saver_dir, filename)
        conditional_director_creator(path)
        self.synthesize(debug_print=False,
                        saver_path=path)

    def synthesize(self, debug_print=False, prog_mapper_print=False, saver_path=None):

        # graphviz_path = os.path.join(saver_path, 'ast_graphs')
        for i, ast_nodes in enumerate(self.ast_nodes):
            beam_js, beam_javas = list(), list()
            if prog_mapper_print:
                print(i)
                # [candy.debug_print(self.config.vocab.chars_type) for candy in ast_nodes]
                print("--------------------------------")
                self.prog_mapper.decode_paths(i, partial=False)
                print("--------------------------------")

            if debug_print:
                print(i)
                print("----------------AST-------------------")
                for ast_path in self.ast_paths[i]:
                    print([item[0] for item in ast_path])
                print("----------------FP-------------------")
                for fp_path in self.fp_paths[i]:
                    print([item[0] for item in fp_path])
                print("----------------Ret Type------------------")
                print(self.ret_types[i])
                print(' ========== done ==========\n\n')

            for j, ast_node in enumerate(ast_nodes):
                # path = os.path.join(graphviz_path, 'program-ast-' + str(i) + 'beam-' + str(j) + '.gv')
                # self.ast_visualizer.visualize_from_ast_head(ast_node.head, ast_node.log_probability, save_path=path)

                ast_dict = {}
                _js = self.json_synthesis.paths_to_ast(ast_node.head)
                ast_dict['program'] = _js
                ret_type = self.config.vocab.chars_type[ast_node.return_type]
                ast_dict['return_type'] = ret_type
                fps = []
                for fp in ast_node.formal_param_inputs:
                    fp_ = self.config.vocab.chars_type[fp]
                    if fp_ not in [DELIM]:
                        fps.append(fp_)
                ast_dict['formal_params'] = fps
                java_program = self.java_synthesis.program_synthesize_from_json(ast_dict)
                beam_js.append(ast_dict)
                beam_javas.append(java_program)
            self.javas_synthesized.append(beam_javas)
            self.jsons_synthesized.append({'programs': beam_js})
        dump_json(self.jsons_synthesized, os.path.join(saver_path, 'beam_search_programs.json'))
        dump_java(self.javas_synthesized, os.path.join(saver_path, 'beam_search_programs.java'))

    def test_viability(self, folder=None, filename=None, debug_print=True):
        print('Doing a viability test')
        total, passed_count, void_count, undeclared_var_count, type_mismatch_count = 0, 0, 0, 0, 0
        for ast_nodes in self.ast_nodes:
            for ast_node in ast_nodes:
                head_node = ast_node.head
                passed = False

                try:
                    self.ast_checker.check_generated_progs(head_node)
                    passed = True
                except VoidProgramException:
                    void_count += 1
                    undeclared_var_count += 1
                    type_mismatch_count += 1
                except UndeclaredVarException:
                    undeclared_var_count += 1
                    type_mismatch_count += 1
                except TypeMismatchException:
                    type_mismatch_count += 1

                passed_count += 1 if passed else 0
                total += 1

        output  = '{:8d} programs/asts in total\n'.format(total)
        output += '{:8d} programs/asts missed for being void\n'.format(void_count)
        output += '{:8d} programs/asts missed for void/illegal var access\n'.format(undeclared_var_count)
        output += '{:8d} programs/asts missed for void/illegal var access/type mismatch\n'.format(type_mismatch_count)
        output += '{:8d} programs/asts passed\n'.format(passed_count)
        path = os.path.join(self.saver_dir, folder)
        conditional_director_creator(path)
        path = os.path.join(path, filename + '.txt')
        if debug_print:
            print(output)
        dump_file(output, path)


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python_recursion_limit', type=int, default=1000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--continue_from', type=str, default='save',
                        help='ignore config options and continue training model checkpointed here')
    parser.add_argument('--data', default='../data_extraction/data_reader/data')
    parser.add_argument('--saver', type=str, default='plots/test_model/')

    clargs = parser.parse_args()
    clargs.saver = os.path.join(clargs.saver, clargs.continue_from)
    conditional_director_creator(clargs.saver)
    sys.setrecursionlimit(clargs.python_recursion_limit)
    memory_test = MasterTester(model_path=clargs.continue_from,
                               save_path=clargs.saver,
                               data_path=clargs.data,
                               beam_width=5,
                               seed=500)
    memory_test.get_states()
    memory_test.plot_tsne(filename='tSNE')
    memory_test.plot_jaccard(filename='jaccard')
    memory_test.get_memory_programs(filename='memory', )
    memory_test.test_viability(folder='viability', filename='memory_viability')
    memory_test.get_random_programs(filename='random')
    memory_test.test_viability(folder='viability', filename='random_viability')

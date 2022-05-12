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

import warnings
from tqdm import tqdm
import numpy as np
import os

from data_extraction.data_reader.data_reader import Reader
from program_helper.ast.parser.ast_gen_checker import AstGenChecker
from program_helper.ast.parser.ast_similarity_checker import AstSimilarityChecker
from program_helper.ast.parser.ast_traverser import AstTraverser
from program_helper.program_beam_searcher import ProgramBeamSearcher
from synthesis.json_synthesis import JSON_Synthesis
from synthesis.write_java import Write_Java
from trainer_vae.infer import BayesianPredictor
from data_extraction.data_reader.data_loader import Loader
from program_helper.program_reverse_map import ProgramRevMapper
from utilities.basics import dump_json, dump_java, stripJavaDoc
from utilities.logging import create_logger

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize


class InferModelHelper:
    '''
        @filepath controls whether you are reading from a new file or not.
        if unused (i.e. set to None) it reduces to a memory test from the training data
    '''

    def __init__(self, model_path=None,
                 seed=0,
                 beam_width=5,
                 max_num_data=1000,
                 depth='change',
                 visibility=1.00,
                 ):

        self.model_path = model_path
        self.infer_model = BayesianPredictor(model_path, batch_size=beam_width, seed=seed, depth=depth)
        self.prog_mapper = ProgramRevMapper(self.infer_model.config.vocab)
        self.visibility = visibility

        self.max_num_data = max_num_data
        self.max_batches = max_num_data // beam_width

        self.reader = None
        self.loader = None

        self.program_beam_searcher = ProgramBeamSearcher(self.infer_model)
        self.logger = create_logger(os.path.join(model_path, 'test_ast_checking.log'))
        self.ast_checker = AstGenChecker(self.program_beam_searcher.infer_model.config.vocab,
                                         compiler=None,
                                         logger=self.logger)

        self.ast_sim_checker = AstSimilarityChecker(logger=self.logger)

        self.json_synthesis = JSON_Synthesis()
        self.java_synthesis = Write_Java()

    def read_and_dump_data(self, filepath=None,
                           data_path=None,
                           min_num_data=0,
                           repair_mode=True,
                           reader_dumps_ast=False):
        '''
        reader will dump the new data into the data path
        '''
        self.reader = Reader(
            dump_data_path=data_path,
            infer=True,
            infer_vocab_path=os.path.join(self.model_path, 'config.json'),
            repair_mode=repair_mode,
            dump_ast=reader_dumps_ast
        )
        self.reader.read_file(filename=filepath,
                              max_num_data=self.max_num_data)
        self.reader.wrangle(min_num_data=min_num_data)
        self.reader.log_info()
        self.reader.dump()

    def get_next_token(self, data_path=None,
                       categories=None
                       ):

        if categories is None:
            categories = ['concept', 'api', 'type', 'clstype', 'var', 'op', 'method']

        self.loader = Loader(data_path, self.infer_model.config)

        avg_prob = self.get_avg_api_prob(categories=categories)
        return avg_prob

    def get_horizon_prob(self, data_path=None,
                         ignore_concepts=True
                         ):

        self.loader = Loader(data_path, self.infer_model.config)

        avg_prob_dict = self._get_horizon_prob(ignore_concepts=ignore_concepts)

        for key in avg_prob_dict.keys():
            count = avg_prob_dict[key][0]
            value = avg_prob_dict[key][1]
            real_value = avg_prob_dict[key][2]

            avg_prob_dict[key] = (value / count, real_value / count)
        return avg_prob_dict

    def synthesize_programs(self, data_path=None,
                            debug_print=False,
                            viability_check=True,
                            dump_result_path=None,
                            dump_jsons=False,
                            dump_psis=False,
                            max_programs=None,
                            real_ast_jsons=None
                            ):

        self.loader = Loader(data_path, self.infer_model.config)
        ## TODO: need to remove
        self.ast_checker.java_compiler = self.loader.program_reader.java_compiler

        psis, mappers, method_embeddings = self.encode_inputs()

        jsons_synthesized, javas_synthesized = self.decode_programs(
            initial_state=psis,
            debug_print=debug_print,
            viability_check=viability_check,
            max_programs=max_programs,
            real_ast_jsons=self.loader.program_reader.json_programs["programs"],
            mappers=mappers,
            method_embeddings=method_embeddings
        )
        # real_codes = self.extract_real_codes(self.loader.program_reader.json_programs)

        # self.program_beam_searcher.tree_beam_searcher.print_infer_horizon_stat()
        if dump_result_path is not None:
            dump_java(javas_synthesized, os.path.join(dump_result_path, 'beam_search_programs.java')
                      )

            if dump_jsons is True:
                dump_json(jsons_synthesized, os.path.join(dump_result_path, 'beam_search_programs.json'))

            if dump_psis is True:
                dump_json({'embeddings': [psi.tolist() for psi in psis]},
                          os.path.join(dump_result_path + '/EmbeddedProgramList.json'))

        return

    def get_jaccard_probabilities(self, data_path=None,
                                  debug_print=False,
                                  max_programs=None,
                                  real_ast_jsons=None,
                                  real_javas=None
                                  ):

        self.loader = Loader(data_path, self.infer_model.config)
        ## TODO: need to remove
        self.ast_checker.java_compiler = self.loader.program_reader.java_compiler

        psis, mappers, method_embeddings = self.encode_inputs()

        jsons_synthesized, javas_synthesized = self.decode_programs(
            initial_state=psis,
            debug_print=debug_print,
            viability_check=False,
            max_programs=max_programs,
            real_ast_jsons=self.loader.program_reader.json_programs["programs"],
            mappers=mappers,
            method_embeddings=method_embeddings,
            real_javas=real_javas
        )

        # real_json = self.extract_real_codes(self.loader.program_reader.json_programs)

        return

    def extract_real_codes(self, json_program):
        # real_codes = []
        # for js in json_program['programs']:
        #     real_codes.append(js['body'])
        return json_program

    def encode_inputs(self,
                      skip_batches=0
                      ):

        psis = []
        mappers = []
        method_embeddings = []
        batch_num = 0
        while True:
            try:
                batch = self.loader.next_batch()
                skip_batches -= 1
                if skip_batches >= 0:
                    continue
            except StopIteration:
                break
            psi, all_var_mappers, method_embedding = self.infer_model.get_initial_state_from_next_batch(batch,
                                                                                                        visibility=self.visibility)
            psi_ = np.transpose(np.array(psi), [1, 0, 2])  # batch_first
            psis.extend(psi_)
            mappers.extend(all_var_mappers)
            method_embeddings.extend(method_embedding)
            self.prog_mapper.add_batched_data(batch)
            batch_num += 1
            if batch_num >= self.max_batches:
                break

        return psis, mappers, method_embeddings

    def get_avg_api_prob(self,
                         skip_batches=0,
                         categories=None,
                         ):

        batch_num = 0
        total_prob = [0.] * 7
        total_nodes = [0] * 7
        while True:
            try:
                batch = self.loader.next_batch()
                skip_batches -= 1
                if skip_batches >= 0:
                    continue
            except StopIteration:
                break
            probs = self.infer_model.get_api_prob_from_next_batch(batch, visibility=self.visibility)
            for j, prob in enumerate(probs):
                total_prob[j] += prob[0]  # Output is average over batch
                total_nodes[j] += prob[1]  # Output is average over batch
            batch_num += 1
            if batch_num >= self.max_batches:
                break

        result_dict = dict()
        for j, cat in enumerate(categories):
            result_dict[cat] = total_prob[j] / total_nodes[j]

        result_dict['all'] = sum(total_prob) / sum(total_nodes)
        result_dict['all_but_concept'] = (sum(total_prob) - total_prob[0]) / \
                                         (sum(total_nodes) - total_nodes[0])

        return result_dict

    def _get_horizon_prob(self,
                          skip_batches=0,
                          ignore_concepts=True
                          ):

        batch_num = 0
        prob_horizon = dict()  # each element is length -> [count, cumul_sum]
        while True:
            try:
                batch = self.loader.next_batch()
                skip_batches -= 1
                if skip_batches >= 0:
                    continue
            except StopIteration:
                break
            new_probs = self.infer_model.get_horizon_prob_from_next_batch(batch, visibility=self.visibility, ignore_concepts=ignore_concepts)
            for len in new_probs.keys():
                count, curr_sum, curr_real_sum = new_probs[len]
                if len not in prob_horizon:
                    prob_horizon[len] = [0, 0.0, 0.0]
                prob_horizon[len][0] += count
                prob_horizon[len][1] += curr_sum
                prob_horizon[len][2] += curr_real_sum
            batch_num += 1
            if batch_num >= self.max_batches:
                break

        return prob_horizon

    def decode_programs(self, initial_state=None,
                        debug_print=False,
                        viability_check=True,
                        max_programs=None,
                        real_ast_jsons=None,
                        mappers=None,
                        method_embeddings=None,
                        real_javas=None
                        ):
        jsons_synthesized = list()
        javas_synthesized = list()
        outcome_strings = ['' for _ in range(self.infer_model.config.batch_size)]
        sz = min(max_programs, len(initial_state)) if max_programs is not None else len(initial_state)
        for i in tqdm(range(sz)):
            temp = [initial_state[i] for _ in range(self.infer_model.config.batch_size)]
            ast_nodes = self.program_beam_searcher.beam_search_memory(
                initial_state=temp,
                ret_type=self.prog_mapper.get_return_type(i),
                fp_types=self.prog_mapper.get_fp_type_inputs(i),
                field_types=self.prog_mapper.get_field_types(i),
                surrounding=self.prog_mapper.get_surrounding(i),
                mapper=mappers[i],
                method_embedding=method_embeddings[i]
            )
            method_name = self.prog_mapper.get_reconstructed_method_name(i,
                                                                         vocab=self.infer_model.config.vocab.chars_kw)

            if viability_check:
                fp_type_names, ret_typ_name, field_typ_names = self.prog_mapper.get_fp_ret_and_field_names(i,
                                                                                                           vocab=self.infer_model.config.vocab.chars_type)

                valid_methods_flag = np.sum(method_embeddings[i], axis=1) > 0
                original_surrounding_data = self.prog_mapper.get_surrounding(i)
                original_rets, original_fps, original_method_names = original_surrounding_data
                dropped_surrounding_data = [original_rets * valid_methods_flag,
                                            original_fps * np.expand_dims(valid_methods_flag, axis=1),
                                            original_method_names * np.expand_dims(valid_methods_flag, axis=1)]

                outcome_strings = self.ast_checker.run_viability_check(ast_nodes,
                                                                       ret_type=ret_typ_name,
                                                                       fp_types=fp_type_names,
                                                                       field_vals=field_typ_names,
                                                                       mapper=mappers[i],
                                                                       surrounding=dropped_surrounding_data,
                                                                       debug_print=False
                                                                       )

            beam_js, beam_javas = self.get_json_and_java(
                ast_nodes,
                outcome_strings,
                type_vocab=self.infer_model.config.vocab.chars_type,
                name=method_name,
                mapper=mappers[i]
            )

            jsons_synthesized.append({'programs': beam_js})
            javas_synthesized.append(beam_javas)

            if real_ast_jsons is not None:
                real_ast_json = real_ast_jsons[i]
                self.ast_sim_checker.check_similarity_for_all_beams(real_ast_json, beam_js)

            if debug_print:
                self.debug_print_fn(i, ast_nodes, prog_mapper=self.prog_mapper)

        if viability_check:
            self.ast_checker.print_stats()

        if real_ast_jsons is not None:
            self.ast_sim_checker.print_stats()

        if real_javas is not None:
            self.get_bleu_score(real_javas, javas_synthesized)

        return jsons_synthesized, javas_synthesized

    def get_bleu_score(self, real_javas, javas_synthesized):
        for weight in [(1., 0., 0., 0.), (0.5, 0.5, 0., 0.), (0, 1., 0., 0.), (0.5, 0.25, 0.25, 0.), ]:
            avg_max_bleu_score, avg_bleu_score = InferModelHelper.calculate_bleu_score(real_javas,
                                                                                       javas_synthesized,
                                                                                       weights=weight)
            self.logger.info('\n\tBleu weight :: {}'.format(weight))
            self.logger.info('\tAverage Max Bleu Score :: {0:0.4f}'.format(avg_max_bleu_score))
            self.logger.info('\tAverage Bleu Score :: {0:0.4f}'.format(avg_bleu_score))

    @staticmethod
    def prune_comments(java_code):
        lines = []
        for line in java_code.splitlines():
            if not line.startswith('//'):
                lines.append(line)
        output = '\n'.join(lines)
        return output.strip()

    @staticmethod
    def calculate_bleu_score(real_javas, javas_synthesized, weights=(1.0, 0, 0, 0)):
        warnings.filterwarnings("ignore")
        avg_max_bleu_score = 0.
        avg_bleu_score = 0.
        for real_java, predictions in zip(real_javas, javas_synthesized):
            real_java = stripJavaDoc(real_java)
            real_java = real_java.replace('.', ' ').strip()
            real_java_tokens = word_tokenize(real_java)
            max_bleu = 0.
            for j, prediction in enumerate(predictions):
                prediction = InferModelHelper.prune_comments(prediction).replace('.', ' ').strip()
                prediction_tokens = word_tokenize(prediction)
                bleu = sentence_bleu(real_java_tokens, prediction_tokens, weights=weights)
                if bleu > max_bleu:
                    max_bleu = bleu
                avg_bleu_score += bleu
            avg_max_bleu_score += max_bleu

        avg_max_bleu_score /= len(real_javas)
        avg_bleu_score /= len(real_javas) * len(javas_synthesized[0][0])

        return avg_max_bleu_score, avg_bleu_score

    def get_json_and_java(self, ast_nodes, outcome_strings, name='foo', type_vocab=None, mapper=None):

        beam_js, beam_javas = list(), list()
        for j, (outcome, ast_node) in enumerate(zip(outcome_strings, ast_nodes)):
            # path = os.path.join(saver_path, 'program-ast-' + str(i) + 'beam-' + str(j) + '.gv')
            # ast_visualizer.visualize_from_ast_head(ast_node.head, ast_node.log_probability, save_path=path)

            ast_dict = self.json_synthesis.full_json_extract(ast_node, type_vocab, name=name)
            java_program = self.java_synthesis.program_synthesize_from_json(ast_dict,
                                                                            beam_id=j,
                                                                            comment=outcome,
                                                                            prob=ast_node.log_probability,
                                                                            mapper=mapper
                                                                            )
            beam_js.append(ast_dict)
            beam_javas.append(java_program)
        return beam_js, beam_javas

    def debug_print_fn(self, i, ast_candies, prog_mapper=None):

        print(i)
        if prog_mapper is not None:
            prog_mapper.decode_paths(i)
        print("----------------AST-------------------")
        ast_traverser = AstTraverser()
        ast_paths = [ast_traverser.depth_first_search(candy.head) for candy in ast_candies]
        for ast_path in ast_paths:
            print([item[0] for item in ast_path])

    def close(self):
        self.infer_model.close()

    def reset(self):
        self.prog_mapper.reset()
        self.ast_checker.java_compiler = None
        self.loader = None
        self.reader = None
        self.ast_checker.reset_stats()
        self.ast_sim_checker.reset_stats()

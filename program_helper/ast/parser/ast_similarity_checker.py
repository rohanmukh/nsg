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
import itertools
from itertools import chain

from data_extraction.data_reader.utils import gather_calls, get_paths_topdown
from program_helper.ast.parser.ast_parser_simple import AstParserSimple, get_paths, get_paths_of_calls
from program_helper.set.apicalls import ApiCalls
from collections import defaultdict

from utilities.basics import truncate_two_decimals


class MyDefaultDict:
    def __init__(self, types):
        self.vals = []
        for type in types:
            self.vals.append(defaultdict(type))

    def get_value(self, length):
        out = []
        for _dict in self.vals:
            out.append(_dict[length])
        return out

    def set_value(self, length, v1, v2, v3):
        vals = [v1, v2, v3]
        assert len(vals) == len(self.vals)
        for _dict, _val in zip(self.vals, vals):
            _dict[length] = _val
        return

    def keys(self):
        return self.vals[0].keys()

    def get_item(self, k):
        out_ = []
        for val in self.vals:
            out_.append(val[k])
        return out_

    def get_item_at_id(self, k, id=2):
        return self.vals[id][k]

class AstSimilarityChecker:

    def __init__(self,
                 logger=None):
        self.logger = logger
        self.sum_jaccard = 0.
        self.sum_jaccard_seq_calls = 0.
        self.sum_jaccard_seq = 0.
        self.sum_jaccard_ast = 0.

        self.count = 0
        self.max_jaccard = 0.
        self.min_jaccard = None

        self.max_similarity_count = 0
        self.max_similarity_like_bayou = 0.0
        self.max_similarity_like_bayou_seq = 0.0
        self.max_similarity_like_bayou_seq_calls = 0.0
        self.max_similarity_like_bayou_ast = 0.0


        self.max_similarity_by_prog_length = MyDefaultDict((int, float, float))
        self.simple_parser = AstParserSimple()
        return

    def reset_stats(self):
        self.max_similarity_count = 0
        self.max_similarity_like_bayou = 0.0
        self.max_similarity_like_bayou_ast = 0.0
        self.sum_jaccard = 0.
        self.sum_jaccard_ast = 0.
        self.count = 0
        self.max_jaccard = 0.
        self.min_jaccard = 0.

    def check_similarity_for_all_beams(self, real_ast_json, predicted_ast_jsons):
        max_similarity = 0.0
        max_similarity_seq = 0.0
        max_similarity_seq_calls = 0.0
        max_similarity_ast = 0.0
        validity = False
        for pred_ast_json in predicted_ast_jsons:
            similarity, similarity_seq, similarity_seq_calls, similarity_ast, valid = self.check_similarity(real_ast_json, pred_ast_json)
            if valid:
                validity = True
                max_similarity = max(similarity, max_similarity)
                max_similarity_seq = max(similarity_seq, max_similarity_seq)
                max_similarity_seq_calls = max(similarity_seq_calls, max_similarity_seq_calls)
                max_similarity_ast = max(similarity_ast, max_similarity_ast)

        if validity:
            self.update_max_similarity_stat(max_similarity, max_similarity_seq, max_similarity_seq_calls, max_similarity_ast)
            self.update_max_similarity_by_length_stat(max_similarity, length=len(gather_calls(real_ast_json['ast'])))
        return max_similarity

    def update_max_similarity_stat(self, max_similarity, max_similarity_seq, max_similarity_seq_calls, max_similarity_ast):
        self.max_similarity_count += 1
        self.max_similarity_like_bayou += max_similarity
        self.max_similarity_like_bayou_seq += max_similarity_seq
        self.max_similarity_like_bayou_seq_calls += max_similarity_seq_calls
        self.max_similarity_like_bayou_ast += max_similarity_ast

    def update_max_similarity_by_length_stat(self, max_similarity, length):
        curr_count, curr_similarity, curr_avg_similarity = self.max_similarity_by_prog_length.get_value(length)
        new_count, new_similarity = curr_count + 1, curr_similarity + max_similarity
        new_avg_similarity = new_similarity/new_count
        self.max_similarity_by_prog_length.set_value(length, new_count, new_similarity, new_avg_similarity)




    def check_similarity(self, real_ast, pred_ast):
        calls = gather_calls(real_ast['ast'])

        real_ast_parsed = self.simple_parser.form_ast(real_ast['ast']['_nodes'])

        apicalls1 = list(set(chain.from_iterable([ApiCalls.from_call(call)
                                                 for call in calls])))
        real_paths = set(get_paths(real_ast_parsed)) - {()}
        real_paths_calls = set(get_paths_of_calls(real_ast_parsed)) - {()}


        calls = gather_calls(pred_ast['ast'])
        pred_ast_parsed = self.simple_parser.form_ast(pred_ast['ast']['_nodes'])
        pred_paths = set(get_paths(pred_ast_parsed)) - {()}
        pred_paths_calls = set(get_paths_of_calls(pred_ast_parsed)) - {()}


        sim_seq = AstSimilarityChecker.get_jaccard_similarity(real_paths, pred_paths)
        sim_seq_calls = AstSimilarityChecker.get_jaccard_similarity(real_paths_calls, pred_paths_calls)


        apicalls2 = list(set(chain.from_iterable([ApiCalls.from_call(call)
                                                 for call in calls])))


        sim_ast = float(real_ast['ast'] == pred_ast['ast'])

        similarity = AstSimilarityChecker.get_jaccard_similarity(set(apicalls1), set(apicalls2))

        valid = len(apicalls1) + len(apicalls2) > 0
        if valid:
            self.update_statistics(similarity, sim_seq, sim_seq_calls, sim_ast)

        return similarity, sim_seq, sim_seq_calls, sim_ast, valid

    @staticmethod
    def get_jaccard_similarity(setA, setB):

        if (len(setA) == 0) and (len(setB) == 0):
            return 1

        setA = set(setA)
        setB = set(setB)

        sim = len(setA & setB) / len(setA | setB)
        return sim

    def update_statistics(self, curr_similarity, curr_similarity_seq, sim_seq_calls, curr_similarity_ast):
        self.sum_jaccard += curr_similarity
        self.sum_jaccard_seq += curr_similarity_seq
        self.sum_jaccard_seq_calls += sim_seq_calls
        self.sum_jaccard_ast += curr_similarity_ast
        self.count += 1
        if curr_similarity > self.max_jaccard:
            self.max_jaccard = curr_similarity
        if self.min_jaccard is None or curr_similarity < self.min_jaccard:
            self.min_jaccard = curr_similarity


    def print_stats(self):
        avg_similarity = self.sum_jaccard / (self.count + 0.00001)
        avg_similarity_seq_call = self.sum_jaccard_seq_calls / (self.count + 0.00001)
        avg_similarity_seq= self.sum_jaccard_seq / (self.count + 0.00001)
        avg_similarity_ast = self.sum_jaccard_ast / (self.count + 0.00001)
        self.logger.info('')
        self.logger.info('\tAverage Jaccard Similarity :: {0:0.4f}'.format( avg_similarity))
        self.logger.info('\tAverage Jaccard Similarity Seq call :: {0:0.4f}'.format( avg_similarity_seq_call))
        self.logger.info('\tAverage Jaccard Similarity Seq :: {0:0.4f}'.format( avg_similarity_seq))
        self.logger.info('\tAverage Jaccard Similarity ast :: {0:0.4f}'.format( avg_similarity_ast))
        # self.logger.info('\tMaximum Jaccard Similarity :: {0:0.4f}'.format(self.max_jaccard))
        # self.logger.info('\tMinimum Jaccard Similarity :: {0:0.4f}'.format(self.min_jaccard))

        avg_max_bayou = self.max_similarity_like_bayou / (self.max_similarity_count + 0.00001)
        avg_max_bayou_seq_calls = self.max_similarity_like_bayou_seq_calls / (self.max_similarity_count + 0.00001)
        avg_max_bayou_seq = self.max_similarity_like_bayou_seq / (self.max_similarity_count + 0.00001)
        avg_max_bayou_ast = self.max_similarity_like_bayou_ast / (self.max_similarity_count + 0.00001)
        self.logger.info('\tMaximum Jaccard Similarity amongst all beams :: {0:0.4f}'.format(avg_max_bayou))
        self.logger.info('\tMaximum Jaccard Similarity seq calls amongst all beams :: {0:0.4f}'.format(avg_max_bayou_seq_calls))
        self.logger.info('\tMaximum Jaccard Similarity seq amongst all beams :: {0:0.4f}'.format(avg_max_bayou_seq))
        self.logger.info('\tMaximum Jaccard Similarity ast amongst all beams :: {0:0.4f}'.format(avg_max_bayou_ast))

        keys = self.max_similarity_by_prog_length.keys()
        for k in sorted(keys):
            val = self.max_similarity_by_prog_length.get_item_at_id(k, id=2)
            count = self.max_similarity_by_prog_length.get_item_at_id(k, id=0)
            print("Length of program {} :: Average Maximum Jaccard Similarity amongst all beams :: {} count :: {}".
                  format(k, truncate_two_decimals(val), count))

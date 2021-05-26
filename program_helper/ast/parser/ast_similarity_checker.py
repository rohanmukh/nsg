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
from copy import deepcopy
from itertools import chain

from data_extraction.data_reader.utils import gather_calls
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
        self.count = 0
        self.max_jaccard = 0.
        self.min_jaccard = None

        self.min_distance_count = 0
        self.min_distance_like_bayou = 0.0

        self.min_distance_by_prog_length = MyDefaultDict((int, float, float))
        return

    def reset_stats(self):
        self.min_distance_count = 0
        self.min_distance_like_bayou = 0.0
        self.sum_jaccard = 0.
        self.count = 0
        self.max_jaccard = 0.
        self.min_jaccard = 0.

    def check_similarity_for_all_beams(self, real_ast_json, predicted_ast_jsons):
        min_distance = 1.0
        for pred_ast_json in predicted_ast_jsons:
            distance = 1 - self.check_similarity(real_ast_json, pred_ast_json)
            min_distance = min(distance, min_distance)

        self.update_min_distance_stat(min_distance)

        self.update_min_distance_by_length_stat(min_distance, length=len(gather_calls(real_ast_json['ast'])))
        return min_distance

    def update_min_distance_stat(self, min_distance):
        self.min_distance_count += 1
        self.min_distance_like_bayou += min_distance

    def update_min_distance_by_length_stat(self, min_distance, length):
        curr_count, curr_distance, curr_avg_distance = self.min_distance_by_prog_length.get_value(length)
        new_count, new_distance = curr_count + 1, curr_distance + min_distance
        new_avg_distance = new_distance/new_count
        self.min_distance_by_prog_length.set_value(length, new_count, new_distance, new_avg_distance)


    def check_similarity(self, real_ast, pred_ast):
        calls = gather_calls(real_ast['ast'])
        apicalls1 = list(set(chain.from_iterable([ApiCalls.from_call(call)
                                                 for call in calls])))

        calls = gather_calls(pred_ast['ast'])
        apicalls2 = list(set(chain.from_iterable([ApiCalls.from_call(call)
                                                 for call in calls])))

        distance = AstSimilarityChecker.get_jaccard_similarity(set(apicalls1), set(apicalls2))
        self.update_statistics(distance)
        return distance

    @staticmethod
    def get_jaccard_similarity(setA, setB):

        if (len(setA) == 0) and (len(setB) == 0):
            return 0

        setA = set(setA)
        setB = set(setB)

        sim = len(setA & setB) / len(setA | setB)
        return sim

    def update_statistics(self, curr_distance):
        self.sum_jaccard += curr_distance
        self.count += 1
        if curr_distance > self.max_jaccard:
            self.max_jaccard = curr_distance
        if self.min_jaccard is None or curr_distance < self.min_jaccard:
            self.min_jaccard = curr_distance


    def print_stats(self):
        avg_distance = self.sum_jaccard / (self.count + 0.00001)
        self.logger.info('')
        self.logger.info('\tAverage Jaccard Similarity :: {0:0.4f}'.format( 1 - avg_distance))
        # self.logger.info('\tMaximum Jaccard Similarity :: {0:0.4f}'.format(self.max_jaccard))
        # self.logger.info('\tMinimum Jaccard Similarity :: {0:0.4f}'.format(self.min_jaccard))

        avg_min_bayou = self.min_distance_like_bayou / (self.min_distance_count + 0.00001)
        self.logger.info('\tMaximum Jaccard Similarity amongst all beams :: {0:0.4f}'.format(1-avg_min_bayou))

        keys = self.min_distance_by_prog_length.keys()
        for k in sorted(keys):
            val = self.min_distance_by_prog_length.get_item_at_id(k, id=2)
            count = self.min_distance_by_prog_length.get_item_at_id(k, id=0)
            print("Length of program {} :: Average Maximum Jaccard Similarity amongst all beams :: {} count :: {}".
                  format(k, truncate_two_decimals(1-val), count))

# Copyright 2017 Rice UniversityDAPIInvoke.split_api_call(value2add)
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

from program_helper.ast.ops import CHILD_EDGE, Node, DSubTree, DVarDecl, DType, DStop
from synthesis.ops.candidate_ast import CONCEPT_NODE, TYPE_NODE
from utilities.vocab_building_dictionary import DELIM
import numpy as np


class Candidate:
    def __init__(self, initial_state, ret_type, prob):
        ## SYNTHESIS
        self.head = self.tree_currNode = DSubTree()

        self.return_type = ret_type

        self.curr_node_val = self.head.val
        self.curr_edge = CHILD_EDGE
        self.next_node_type = CONCEPT_NODE
        self.control_flow_stack = []
        ## DEBUGGING PURPOSE
        self.storage = []

        ## BEAM SEARCH PURPOSES
        self.length = 1
        self.log_probability = -np.inf if prob is None else prob
        self.rolling = True
        self.state = initial_state


    def is_rolling(self):
        return self.rolling

    def is_not_rolling(self):
        return not self.is_rolling()

    def stop_rolling(self):
        self.rolling = False
        return

    def length_mod_and_check(self, curr_val, max_length):
        self.length += 1 \
                       if self.next_node_type == CONCEPT_NODE \
                       and curr_val in [DVarDecl.name()] \
                       else 0
        if self.length >= max_length:
            self.stop_rolling()
        return

    def add_to_storage(self):
        self.storage.append([self.curr_node_val])

    def debug_print(self, vocab_types):
        for i in range(len(self.storage[0])):
            for val in self.storage:
                print(vocab_types[val[i]], end=',')
            print()

    def force_finish(self):
        # TODO
        pass

    def add_node(self, value2add):
        node = self.resolve_node_type(value2add)
        self.tree_currNode = self.tree_currNode.add_and_progress_node(node,
                                                                      edge=self.curr_edge)
        return self

    def resolve_node_type(self, value):
        if value == DVarDecl.name():
            node = DVarDecl({}, {})
        elif self.next_node_type == TYPE_NODE:
            node = DType(value)
        elif value == DSubTree.name():
            node = DSubTree()
        elif value == DStop.name():
            node = DStop()
        else:
            # print(
            #     "Unknown node generated in FP :: value is " + str(value) +
            #     " next node type is  " + str(self.next_node_type))
            node = DStop()
        return node

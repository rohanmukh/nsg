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
import numpy as np
from copy import deepcopy

from synthesis.ops.candidate import Candidate
from program_helper.ast.ops import Node, DStop, SIBLING_EDGE, CHILD_EDGE, DVarDecl, DSubTree
from synthesis.ops.candidate_ast import CONCEPT_NODE, TYPE_NODE
from synthesis.ops.candidate_element import CandidateElement

MAX_GEN_UNTIL_STOP = 10


class SeqBeamSearcher:

    def __init__(self, infer_model):
        self.infer_model = infer_model
        self.beam_width = infer_model.config.batch_size
        return

    def resolve_curr_node(self, candies):
        inputs = []
        for candy in candies:
            val = self.infer_model.config.vocab.concept_dict[candy.curr_node_val]
            inputs.append([val])
        return inputs

    def resolve_next_node(self, candies, beam_ids):
        next_nodes = []
        for k, candy in enumerate(candies):
            next_node_type = candy.next_node_type
            beam = beam_ids[next_node_type][k]
            if next_node_type == CONCEPT_NODE:
                temp = [self.infer_model.config.vocab.chars_concept[idx] for idx in beam]
            elif next_node_type == TYPE_NODE:
                temp = [self.infer_model.config.vocab.chars_type[idx] for idx in beam]
            else:
                raise Exception
            next_nodes.append(temp)
        return next_nodes

    def resolve_beam_probs(self, candies, beam_ln_probs):

        beam_ln_probs = [[0.0 if j == 0 else -np.inf for j in range(len(candies))]
                         if candy.is_not_rolling()
                         else beam_ln_probs[candy.next_node_type][k] for k, candy in enumerate(candies)]

        new_probs = np.array([candy.log_probability for candy in candies])[:, None] + beam_ln_probs

        length = np.array([candy.length for candy in candies])
        len_norm_probs = new_probs / np.power(length[:, None], 0.0)  # # TURNED OFF, set at 0.0, should be 1.0

        return len_norm_probs

    def beam_search(self, initial_state=None, ret_type=None, probs=None):

        candies = [Candidate(initial_state[k], ret_type[k], probs[k]) for k in range(self.beam_width)]

        while not self.check_for_all_stop(candies):
            candies = self.get_next_output_with_fan_out(candies)

        [candy.force_finish() for candy in candies]
        candies.sort(key=lambda x: x.log_probability, reverse=True)
        return candies

    def check_for_all_stop(self, candies):
        return all([not candy.is_rolling() for candy in candies])

    def get_next_output_with_fan_out(self, candies):

        curr_node = self.resolve_curr_node(candies)
        # last_edge = [[candy.curr_edge] for candy in candies]
        states = np.transpose(np.array([candy.state for candy in candies]), [1, 0, 2])

        # states is still topK * LSTM_Decoder_state_size
        # next_node is topK * topK
        # node_probs in  topK * topK
        # log_probabilty is topK
        # [candy.add_to_storage() for candy in candies]
        states, beam_ids, beam_ln_probs = self.infer_model.get_next_seq_state(curr_node,
                                                                              states,
                                                                              candies)
        next_nodes = self.resolve_next_node(candies, beam_ids)
        beam_ln_probs = self.resolve_beam_probs(candies, beam_ln_probs)

        top_k = len(candies)
        rows, cols = np.unravel_index(np.argsort(beam_ln_probs, axis=None)[::-1], (top_k, top_k))
        rows, cols = rows[:top_k], cols[:top_k]

        # rows mean which of the original candidate was finally selected
        new_candies = []
        for row, col in zip(rows, cols):
            new_candy = deepcopy(candies[row])  # candies[row].copy()
            if new_candy.is_rolling():
                new_candy.state = [states[k][row] for k in range(len(states))]
                new_candy.log_probability = beam_ln_probs[row][col]
                value2add = next_nodes[row][col]
                new_candy.length_mod_and_check(value2add, MAX_GEN_UNTIL_STOP)
                if new_candy.is_rolling():
                    new_candy = new_candy.add_node(value2add)

                    if new_candy.next_node_type == CONCEPT_NODE:
                        new_candy.curr_node_val = value2add

                    new_candy = self.handle_control_flow(new_candy)

            new_candies.append(new_candy)

        return new_candies

    def handle_control_flow(self, new_candy):
        node_value = new_candy.tree_currNode.val
        # Modify the control flows as appropriate
        if node_value in DVarDecl.name():
            new_candy = self.handle_DVarDecl(new_candy)
        # Delims
        elif node_value == DStop.name():
            new_candy = self.handle_DStop(new_candy)
        elif node_value in [DSubTree.name()]:
            new_candy = self.handle_single_child(new_candy)
        # default concept nodes
        else:
            new_candy = self.handle_leaf_nodes(new_candy)
        return new_candy


    def handle_DVarDecl(self, new_candy):
        # last we will expand the elements after invoke
        new_candy.control_flow_stack.append(CandidateElement(curr_node=new_candy.tree_currNode,
                                                             edge_path=[SIBLING_EDGE],
                                                             next_node_type=CONCEPT_NODE,
                                                             )
                                            )

        # Right now we will evaluate the Dtype branch
        new_candy.curr_edge = CHILD_EDGE
        new_candy.curr_node_val = DVarDecl.name()
        new_candy.next_node_type = TYPE_NODE
        return new_candy

    def handle_single_child(self, new_candy):
        new_candy.curr_node_val = new_candy.tree_currNode.val
        new_candy.curr_edge = CHILD_EDGE
        new_candy.next_node_type = CONCEPT_NODE
        return new_candy

    def handle_DStop(self, new_candy):
        if len(new_candy.control_flow_stack) == 0:
            new_candy.stop_rolling()
        else:
            element = new_candy.control_flow_stack.pop()
            new_candy.tree_currNode = element.get_current_node()

            ## Update the new candy accordingly
            new_candy.curr_edge = element.edge_path[-1]
            if element.get_curr_node_val() is not None:
                new_candy.curr_node_val = element.get_curr_node_val()
            new_candy.next_node_type = element.get_next_node_type()
        return new_candy

    def handle_leaf_nodes(self, new_candy):
        new_candy = self.handle_DStop(new_candy)
        # new_candy.curr_edge = SIBLING_EDGE
        # new_candy.curr_node_val = new_candy.tree_currNode.val
        return new_candy

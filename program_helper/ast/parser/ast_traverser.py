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

from program_helper.ast.ops import CHILD_EDGE, SIBLING_EDGE
from graphviz import Digraph

ROOT_NONTERMINAL = 'Expression'
VARIABLE_NONTERMINAL = 'Variable'
LITERAL_NONTERMINALS = ['IntLiteral', 'CharLiteral', 'StringLiteral']
LAST_USED_TOKEN_NAME = '<LAST TOK>'
#EXPANSION_LABELED_EDGE_TYPE_NAMES = ["Child"]
EXPANSION_LABELED_EDGE_TYPE_NAMES = []
#EXPANSION_UNLABELED_EDGE_TYPE_NAMES = ["Child", "Parent", "NextUse", "NextToken", "NextSibling", "NextSubtree",
#                                       "InheritedToSynthesised"]
EXPANSION_UNLABELED_EDGE_TYPE_NAMES = ["Child", "Parent", "NextSibling",
                                       "InheritedToSynthesised",
                                       "NextToken", "NextUse"]

expansion_unlabeled_edge_types = OrderedDict(
        (name, edge_id) for (edge_id, name) in enumerate(
            n for n in EXPANSION_UNLABELED_EDGE_TYPE_NAMES))


class AstTraverser:

    @staticmethod
    def depth_first_search(head):

        if head is None:
            return []

        buffer = []

        stack = []
        dfs_id = 0
        parent_id = 0
        stack.append((head, parent_id, CHILD_EDGE))

        while len(stack) > 0:
            node, parent_id, edge_type = stack.pop()

            buffer.append((node.val, node.type, node.valid,
                           node.var_decl_id, node.return_reached,
                           parent_id, edge_type,
                           node.expr_type, node.type_helper, node.return_type,
                           node.iattrib))

            if node.sibling is not None:
                stack.append((node.sibling, dfs_id, SIBLING_EDGE))
            if node.child is not None:
                stack.append((node.child, dfs_id, CHILD_EDGE))

            dfs_id += 1

        return buffer


    @staticmethod
    def dfs_travesal_with_edges(head):
        if head is None:
            return []

        # "Downwards" version of a node (depends on parents & pred's). Keys are IDs from symbol expansion record:
        node_to_inherited_id = {}  #c type: Dict[int, int]
        # "Upwards" version of a node (depends on children). Keys are IDs from symbol expansion record:
        node_to_synthesised_id = {}  # type: Dict[int, int]
        node_to_info = {}
        buffer = []
        last_used_var_id = {}  # type: Dict[str, [int]]
        last_terminal_node_id = -1

        stack = []
        node_id = 0
        dfs_id = 0
        stack.append((head, node_id, dfs_id, CHILD_EDGE))
        nodes = []
        edges = []
        eg_schedule = []

        while len(stack) > 0:
            node_edges = {}
            node, last_sibling, parent_id, edge_type = stack.pop()

            node_info = [
                node.val, node.type, node.valid,
                node.var_decl_id, node.return_reached,
                parent_id, edge_type,
                node.expr_type, node.type_helper, node.return_type,
                node.iattrib]
            node_to_info[node_id] = tuple(node_info)
            setattr(node, 'node_id', [node_id])


            #if node.type == 'DSymtabMod':
            #    print(node_id)
            #    print(node_info)

            # Split the node into inherited node and synthesised node.
            if node.child is not None:
                node_name = str(node.val) + '_inherited'
                #node_name = str(node.type) + '_inherited'
                nodes.append(node_name)
                #node_info[0] = node_name
                #buffer.append(tuple(node_info))
                node_id += 1
                node_to_info[node_id] = node_info
                edges.append((node_id - 1, 'InheritedToSynthesised', node_id))
                node_edges['InheritedToSynthesised'] = (node_id - 1, node_id)
                node.node_id.append(node_id)

                node_name = str(node.val) + '_synthesized'
                #node_name = str(node.type) + '_synthesized'
                #node_info[0] = node_name
                nodes.append(node_name)
                buffer.append(tuple(node_info))
            else:
                node_name = str(node.val) + '_inherited'
                #node_name = str(node.type) + '_inherited'
                #node_name = node.type
                nodes.append(node_name)

                buffer.append(tuple(node_info))
                if last_terminal_node_id != -1:
                    edges.append((last_terminal_node_id, 'NextToken', node_id))
                    last_terminal_node_id = node_id
                else:
                    last_terminal_node_id = node_id
                if node.type == 'DVarAccess' or node.type == 'DVarAccessDecl':
                    if node.val in last_used_var_id:
                        edges.append((last_used_var_id[node.val][-1],
                                     "NextUse",
                                     node_id))
                        last_used_var_id[node.val].append(node_id)
                    else:
                        last_used_var_id[node.val] = [node_id]

                # Change here.
                node_id += 1

                node_name = str(node.val) + '_synthesized'
                #node_name = str(node.type) + '_synthesized'
                nodes.append(node_name)

            if node_id > 1:
                if edge_type:
                    if node.child is not None:
                        edges.append((node_id - 3, 'Child', node_id - 1))
                        edges.append((node_id, 'Parent', node_id - 2))
                        node_edges['Child'] = (node_id - 3, node_id - 1)
                        node_edges['Parent'] = (node_id, node_id - 2)
                        if (node_id - 3) in node_to_inherited_id.keys():
                            node_to_inherited_id[node_id - 3].append(node_id - 1)
                        else:
                            node_to_inherited_id[node_id - 3] = [node_id - 1]
                        node_to_synthesised_id[node_id] = node_id - 2
                    else:
                        edges.append((node_id - 3, 'Child', node_id - 1))
                        edges.append((node_id - 1, 'Parent', node_id - 2))
                        node_edges['Child'] = (node_id - 3, node_id - 1)
                        node_edges['Parent'] = (node_id - 1, node_id - 2)
                        if (node_id - 3) in node_to_inherited_id.keys():
                            node_to_inherited_id[node_id - 3].append(node_id - 1)
                        else:
                            node_to_inherited_id[node_id - 3] = [node_id - 1]
                        node_to_synthesised_id[node_id - 1] = node_id - 2
                    #if node.child is not None:
                    #    # Need more thinking
                    #    edges.append((node_id - 3, 'Child', node_id - 1))
                    #    edges.append((node_id, 'Parent', node_id - 2))
                    #    node_edges['Child'] = (node_id - 3, node_id - 1)
                    #    node_edges['Parent'] = (node_id, node_id - 2)
                    #    if (node_id - 3) in node_to_inherited_id.keys():
                    #        node_to_inherited_id[node_id - 3].append(node_id - 1)
                    #    else:
                    #        node_to_inherited_id[node_id - 3] = [node_id - 1]
                    #    node_to_synthesised_id[node_id] = node_id - 2
                    #else:
                    #    edges.append((node_id - 2, 'Child', node_id))
                    #    edges.append((node_id, 'Parent', node_id - 1))
                    #    node_edges['Child'] = (node_id - 2, node_id)
                    #    node_edges['Parent'] = (node_id, node_id - 1)
                    #    #node_to_inherited_id[node_id - 2] = node_id
                    #    node_to_synthesised_id[node_id] = node_id - 1
                    #    if (node_id - 2) in node_to_inherited_id.keys():
                    #        node_to_inherited_id[node_id - 2].append(node_id)
                    #    else:
                    #        node_to_inherited_id[node_id - 2] = [node_id]
                    #    node_to_synthesised_id[node_id] = node_id - 1
                else:
                    sibling_parent = node_to_synthesised_id[last_sibling]
                    if node.child is not None:
                        # node_to_synthesised_id[last_sibling] always consist
                        # two nodes.
                        edges.append((sibling_parent - 1, 'Child', node_id - 1))
                        edges.append((last_sibling, 'NextSibling', node_id - 1))
                        node_edges['Child'] = (sibling_parent - 1, node_id - 1)
                        node_edges['NextSibling'] = (last_sibling, node_id - 1)
                        if (sibling_parent - 1) in node_to_inherited_id.keys():
                            node_to_inherited_id[sibling_parent - 1].append(node_id - 1)
                        else:
                            node_to_inherited_id[sibling_parent - 1] = [node_id - 1]


                        edges.append((node_id, 'Parent', sibling_parent))
                        node_edges['Parent'] = (node_id, sibling_parent)
                        node_to_synthesised_id[node_id] = sibling_parent
                    else:
                        edges.append((sibling_parent - 1, 'Child', node_id - 1))
                        edges.append((last_sibling, 'NextSibling', node_id - 1))
                        node_edges['Child'] = (sibling_parent - 1, node_id - 1)
                        node_edges['NextSibling'] = (last_sibling, node_id - 1)
                        if (sibling_parent - 1) in node_to_inherited_id.keys():
                            node_to_inherited_id[sibling_parent - 1].append(node_id - 1)
                        else:
                            node_to_inherited_id[sibling_parent - 1] = [node_id - 1]
                    #if node.child is not None:
                    #    # node_to_synthesised_id[last_sibling] always consist
                    #    # two nodes.
                    #    edges.append((sibling_parent - 1, 'Child', node_id - 1))
                    #    edges.append((last_sibling, 'NextSibling', node_id - 1))
                    #    node_edges['Child'] = (sibling_parent - 1, node_id - 1)
                    #    node_edges['NextSibling'] = (last_sibling, node_id - 1)
                    #    if (sibling_parent - 1) in node_to_inherited_id.keys():
                    #        node_to_inherited_id[sibling_parent - 1].append(node_id - 1)
                    #    else:
                    #        node_to_inherited_id[sibling_parent - 1] = [node_id - 1]
                    #else:
                    #    edges.append((sibling_parent - 1, 'Child', node_id))
                    #    edges.append((last_sibling, 'NextSibling', node_id))
                    #    node_edges['Child'] = (sibling_parent - 1, node_id)
                    #    node_edges['NextSibling'] = (last_sibling, node_id)
                    #    if (sibling_parent - 1) in node_to_inherited_id.keys():
                    #        node_to_inherited_id[sibling_parent - 1].append(node_id)
                    #    else:
                    #        node_to_inherited_id[sibling_parent - 1] = [node_id]
                        edges.append((node_id - 1, 'Parent', sibling_parent))
                        node_edges['Parent'] = (node_id - 1, sibling_parent)
                        node_to_synthesised_id[node_id - 1] = sibling_parent

            eg_schedule.append(node_edges)
            if node.sibling is not None:
                if node.child is not None:
                    stack.append((node.sibling, node_id, dfs_id, SIBLING_EDGE))
                else:
                    stack.append((node.sibling, node_id - 1, dfs_id, SIBLING_EDGE))
            else:
                last_sibling = None
            if node.child is not None:
                stack.append((node.child, node_id, dfs_id, CHILD_EDGE))

            node_id += 1
            dfs_id += 1

        #results = [node_to_info, node_to_inherited_id, node_to_synthesised_id, edges]

        #edge_types = ['INHERITED_TO_SYNTHESISED', 'PARENT', 'CHILD', 'NEXTSibling']
        #total_edge_types = len(edge_types)
        #step_by_edge = [[] for _ in range(total_edge_types)]
        return edges, eg_schedule, node_to_inherited_id,\
            node_to_synthesised_id, nodes, buffer


    def brockschmidt_traversal(head, edges, node_to_inherited_id,
                               node_to_synthesised_id):
        if head is None:
            return []

        #result_holder['node_info'] = [head_info]
        eg_schedule = []
        incoming_edges = {}
        for (source, edge_type, target) in edges:
            if target not in incoming_edges.keys():
                incoming_edges[target] = {}
            incoming_edges[target][edge_type] = source

        def expand_node(root_node_id):
            if root_node_id not in node_to_inherited_id.keys():
                return
            child_nodes_id = node_to_inherited_id[root_node_id]
            last_sibling = None
            parent_inwards_edges = defaultdict(list)
            parent_inwards_edges['InheritedToSynthesised'].append(
                (root_node_id, root_node_id + 1))
            for child_id in child_nodes_id:
                child_inwards_edges = defaultdict(list)
                if child_id in incoming_edges.keys():
                    for (edge_type, source) in incoming_edges[child_id].items():
                        child_inwards_edges[edge_type].append((source, child_id))
                eg_schedule.append(child_inwards_edges)
                expand_node(child_id)
                if child_id in node_to_inherited_id.keys():
                    parent_inwards_edges['Parent'].append(
                        (child_id + 1, node_to_synthesised_id[child_id + 1]))
                else:
                    parent_inwards_edges['Parent'].append(
                        (child_id, node_to_synthesised_id[child_id]))
            eg_schedule.append(parent_inwards_edges)
        expand_node(0)

        def split_schedult_step(step):
            total_edge_types = len(EXPANSION_LABELED_EDGE_TYPE_NAMES) + len(
                EXPANSION_UNLABELED_EDGE_TYPE_NAMES)
            step_by_edge = [[] for _ in range(total_edge_types)]
            for (label, edges) in step.items():
                step_by_edge[expansion_unlabeled_edge_types[label]] = edges
            return step_by_edge

        eg_schedule = [split_schedult_step(step) for step in eg_schedule]
        return eg_schedule


    # TODO(ywen666): Need to move out of here.
    def calculate_gnn_info(eg_schedule):
        total_edge_types = len(EXPANSION_UNLABELED_EDGE_TYPE_NAMES)
        eg_propagation_substeps = 100
        eg_initial_node_ids = []
        eg_sending_node_ids = [[[] for _ in range(
            total_edge_types)] for _ in range(eg_propagation_substeps)]
        next_step_target_node_id = [0 for _ in range(eg_propagation_substeps)]
        eg_msg_target_node_ids = [[[] for _ in range(total_edge_types)]
                                  for _ in range(eg_propagation_substeps)]
        eg_receiving_node_ids = [[] for _ in range(eg_propagation_substeps)]
        eg_receiving_node_nums = [0 for _ in range(eg_propagation_substeps)]


        for (step_num, schedule_step) in enumerate(eg_schedule):
            eg_node_id_to_step_target_id = OrderedDict()
            for edge_type in range(total_edge_types):
                for (source, target) in schedule_step[edge_type]:
                    eg_sending_node_ids[step_num][edge_type].append(source)
                    step_target_id = eg_node_id_to_step_target_id.get(target)
                    if step_target_id is None:
                        step_target_id = next_step_target_node_id[step_num]
                        next_step_target_node_id[step_num] += 1
                        eg_node_id_to_step_target_id[target] = step_target_id
                    eg_msg_target_node_ids[step_num][edge_type].append(step_target_id)
            for eg_target_node_id in eg_node_id_to_step_target_id.keys():
                eg_receiving_node_ids[step_num].append(eg_target_node_id)
            eg_receiving_node_nums[step_num] += len(eg_node_id_to_step_target_id)

        info = (eg_sending_node_ids, eg_msg_target_node_ids,
                eg_receiving_node_ids, eg_receiving_node_nums)
        return info

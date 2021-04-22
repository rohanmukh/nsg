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

from program_helper.ast.ops import CHILD_EDGE, SIBLING_EDGE
from graphviz import Digraph

ROOT_NONTERMINAL = 'Expression'
VARIABLE_NONTERMINAL = 'Variable'
LITERAL_NONTERMINALS = ['IntLiteral', 'CharLiteral', 'StringLiteral']
LAST_USED_TOKEN_NAME = '<LAST TOK>'
EXPANSION_LABELED_EDGE_TYPE_NAMES = ["Child"]
EXPANSION_UNLABELED_EDGE_TYPE_NAMES = ["Parent", "NextUse", "NextToken", "NextSibling", "NextSubtree",
                                       "InheritedToSynthesised"]

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

            buffer.append((node.val, node.type, parent_id,
                           edge_type))
            #buffer.append((node.val, node.type, node.valid,
            #               node.var_decl_id, node.return_reached,
            #               parent_id, edge_type,
            #               node.expr_type, node.type_helper, node.return_type,
            #               node.iattrib))

            if node.sibling is not None:
                stack.append((node.sibling, dfs_id, SIBLING_EDGE))
            if node.child is not None:
                stack.append((node.child, dfs_id, CHILD_EDGE))

            dfs_id += 1

        return buffer


    @staticmethod
    def breadth_first_search(head):

        if head is None:
            return []

        buffer = []

        queue = []
        bfs_id = 0
        parent_id = 0
        queue.append((head, parent_id, CHILD_EDGE))

        while len(queue) > 0:
            node, parent_id, edge_type = queue.pop(0)
            buffer.append((node.val, node.type, parent_id,
                           edge_type))
            temp_node = node.sibling
            while temp_node is not None:
                queue.append((node.sibling, bfs_id, SIBLING_EDGE))
            if node.child is not None:
                queue.append((node.child, bfs_id, CHILD_EDGE))
            bfs_id += 1
        return buffer


    @staticmethod
    def my_travesal(head):
        if head is None:
            return []

        # "Downwards" version of a node (depends on parents & pred's). Keys are IDs from symbol expansion record:
        node_to_inherited_id = {}  #c type: Dict[int, int]
        # "Upwards" version of a node (depends on children). Keys are IDs from symbol expansion record:
        node_to_synthesised_id = {}  # type: Dict[int, int]
        node_to_info = {}
        # Maps variable name to the id of the node where it was last used. Keys are variable names, values are from fresh space next to symbol expansion record:
        last_used_node_id = {}  # type: Dict[str, int]

        stack = []
        node_id = 0
        stack.append((head, node_id, CHILD_EDGE))
        nodes = []
        edges = []
        eg_schedule = []

        while len(stack) > 0:
            node_edges = {}
            node, last_sibling, edge_type = stack.pop()

            node_info = (
                node.val, node.type, node.valid,
                node.var_decl_id, node.return_reached,
                last_sibling, edge_type,
                node.expr_type, node.type_helper, node.return_type,
                node.iattrib)
            node_to_info[node_id] = node_info

            # Split the node into inherited node and synthesised node.
            if node.child is not None:
                node_id += 1
                node_to_info[node_id] = node_info
                edges.append((node_id - 1, 'INHERITED_TO_SYNTHESISED', node_id))
                node_edges['INHERITED_TO_SYNTHESISED'] = (node_id - 1, node_id)

            if node_id > 1:
                if edge_type:
                    if node.child is not None:
                        # Need more thinking
                        edges.append((node_id - 3, 'CHILD', node_id - 1))
                        edges.append((node_id, 'PARENT', node_id - 2))
                        node_to_inherited_id[node_id - 3] = node_id - 1
                        node_to_synthesised_id[node_id] = node_id - 2
                        node_edges['CHILD'] = (node_id - 3, node_id - 1)
                        node_edges['PARENT'] = (node_id, node_id - 2)
                    else:
                        edges.append((node_id - 2, 'CHILD', node_id))
                        edges.append((node_id, 'PARENT', node_id - 1))
                        node_to_inherited_id[node_id - 2] = node_id
                        node_to_synthesised_id[node_id] = node_id - 1
                else:
                    sibling_parent = node_to_synthesised_id[last_sibling]
                    if node.child is not None:
                        # node_to_synthesised_id[last_sibling] always consist
                        # two nodes.
                        edges.append((sibling_parent - 1, 'CHILD', node_id - 1))
                        edges.append((last_sibling, 'NEXTSibling', node_id - 1))
                        node_to_inherited_id[sibling_parent - 1] = node_id - 1
                    else:
                        edges.append((sibling_parent - 1, 'CHILD', node_id))
                        edges.append((last_sibling, 'NEXTSibling', node_id))
                        node_to_inherited_id[sibling_parent - 1] = node_id
                    edges.append((node_id, 'PARENT', sibling_parent))
                    node_to_synthesised_id[node_id] = sibling_parent

            if node.sibling is not None:
                stack.append((node.sibling, node_id, SIBLING_EDGE))
            else:
                last_sibling = None
            if node.child is not None:
                stack.append((node.child, node_id, CHILD_EDGE))

            node_id += 1

        results = [node_to_info, node_to_inherited_id, node_to_synthesised_id, edges]

        edge_dict = {}
        for edge in edges:
            if edge in edge_dict.keys():
                edge_dict[edge]

        edge_types = ['INHERITED_TO_SYNTHESISED', 'PARENT', 'CHILD', 'NEXTSibling']
        total_edge_types = len(edge_types)
        step_by_edge = [[] for _ in range(total_edge_types)]
        return results

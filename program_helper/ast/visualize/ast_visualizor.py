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

from graphviz import Digraph
import json

from program_helper.ast.ops import CHILD_EDGE, SIBLING_EDGE


class AstVisualizer:

    def __init__(self):
        self.dot = Digraph(comment='Program AST', format='eps')
        self.id = 0
        self.stack = []

    def visualize(self, head, save_path):
        self.add_node(head, str(0))
        _ = self.dfs_with_dot(head, 0)
        self.dot.render(save_path)

    def add_node(self, node, id):
        node_label = node.val
        label = str(node_label)
        label = '\n'.join(label.split('.'))
        self.dot.node(id, label=label)
        return

    def add_edge(self, parent_id, child_id, edge_type):
        if child_id is None:
            return

        label = 'child' if edge_type else 'sibling'
        # label += " / " + str(child_id)
        if str(parent_id) != str(child_id):
            self.dot.edge(str(parent_id), str(child_id), label=label, constraint='true', direction='LR')
        return

    def dfs_with_dot(self, head, id_start):
        if head is None:
            return id_start

        temp = head.child
        i = id_start
        j = 0
        while temp is not None:
            i += 1
            self.add_node(temp, str(i))
            self.add_edge(str(id_start), str(i), CHILD_EDGE if j == 0 else SIBLING_EDGE)
            i = self.dfs_with_dot(temp, i)
            temp = temp.sibling
            j += 1

        return i

    def visualize_from_ast_head(self, head, prob, save_path='temp.gv'):
        self.dot.clear()
        self.id = 0
        self.stack.clear()
        self.visualize(head, save_path)
        return

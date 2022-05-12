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



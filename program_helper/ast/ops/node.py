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
from utilities.vocab_building_dictionary import DELIM

CHILD_EDGE = True
SIBLING_EDGE = False


class Node:
    def __init__(self, val=None, child=None, sibling=None):
        self.val = val
        self.child = child
        self.sibling = sibling
        self.valid = True

        self.var_decl_id = 0
        self.return_reached = False

        self.expr_type = DELIM
        self.return_type = DELIM
        self.type_helper = DELIM

        self.type = Node.name()
        self.iattrib = [0, 0, 0]

    @staticmethod
    def name():
        return "Node"

    def add_and_progress_node(self, node, edge=None):
        if edge == CHILD_EDGE:
            return self.add_and_progress_child_node(node)
        elif edge == SIBLING_EDGE:
            return self.add_and_progress_sibling_node(node)
        else:
            raise Exception

    def add_and_progress_sibling_node(self, node):
        self.sibling = node
        return self.sibling

    def add_and_progress_child_node(self, node):
        self.child = node
        return self.child

    def add_node(self, node, edge=None):
        if edge == CHILD_EDGE:
            return self.add_child_node(node)
        elif edge == SIBLING_EDGE:
            return self.add_sibling_node(node)
        else:
            raise Exception

    def add_sibling_node(self, node):
        self.sibling = node

    def add_child_node(self, node):
        self.child = node

    def progress_node(self, edge=None):
        if edge == CHILD_EDGE:
            return self.progress_child_node()
        elif edge == SIBLING_EDGE:
            return self.progress_sibling_node()
        else:
            raise Exception

    def progress_sibling_node(self):
        return self.sibling

    def progress_child_node(self):
        return self.child

    def is_valid(self):
        return self.valid

# Copyright 2017 Rice UniversityDComp
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
from program_helper.ast.ops import DStop
from program_helper.ast.ops.leaf_ops.DOp import DOp
from program_helper.ast.ops.node import Node

class DInfix(Node):
    def __init__(self, node_js=None, child=None, sibling=None):
        super().__init__(DInfix.name(), child, sibling)

        if node_js is None:
            return

        self.op = node_js['_op']
        self.child = DOp(self.op)

        self.type = self.val

    @staticmethod
    def name():
        return 'DInfix'

    @staticmethod
    def construct_blank_node():
        invoke_node = DInfix()
        invoke_node.child = ">"
        invoke_node.child.sibling = DStop()
        invoke_node.child.sibling.sibling = DStop()
        return invoke_node


##
class DLeft(Node):
    def __init__(self, child=None, sibling=None):
        super().__init__(DLeft.name(), child, sibling)
        self.type = self.val

    @staticmethod
    def name():
        return 'DLeft'

##
class DRight(Node):
    def __init__(self, child=None, sibling=None):
        super().__init__(DRight.name(), child, sibling)
        self.type = self.val

    @staticmethod
    def name():
        return 'DRight'

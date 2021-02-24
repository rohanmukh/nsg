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
from program_helper.ast.ops import DStop
from program_helper.ast.ops.node import Node


class DBranch(Node):
    def __init__(self, child=None, sibling=None):
        super().__init__(DBranch.name(), child, sibling)
        self.type = self.val

    @staticmethod
    def name():
        return 'DBranch'

    @staticmethod
    def construct_blank_node():
        branch_node = DBranch()
        branch_node.child = DStop()
        branch_node.child.sibling = DStop()
        branch_node.child.sibling.sibling = DStop()
        return branch_node

##
class DCond(Node):
    def __init__(self, child=None, sibling=None):
        super().__init__(DCond.name(), child, sibling)
        self.type = self.val

    @staticmethod
    def name():
        return 'DCond'


class DThen(Node):
    def __init__(self, child=None, sibling=None):
        super().__init__(DThen.name(), child, sibling)
        self.type = self.val

    @staticmethod
    def name():
        return 'DThen'


class DElse(Node):
    def __init__(self, child=None, sibling=None):
        super().__init__(DElse.name(), child, sibling)
        self.type = self.val

    @staticmethod
    def name():
        return 'DElse'

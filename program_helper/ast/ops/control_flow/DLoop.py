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

class DLoop(Node):
    def __init__(self, child=None, sibling=None):
        super().__init__( DLoop.name(), child, sibling)
        self.type = self.val

    @staticmethod
    def name():
        return 'DLoop'

    @staticmethod
    def construct_blank_node():
        except_name = DLoop()
        except_name.child = DStop()
        except_name.child.sibling = DStop()
        return except_name

class DBody(Node):
    def __init__(self, child=None, sibling=None):
        super().__init__(DBody.name(), child, sibling)
        self.type = self.val

    @staticmethod
    def name():
        return 'DBody'
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

from program_helper.ast.ops import Node, DAPICall, DStop


class DAPICallSingle(Node):
    def __init__(self, _call=None,
                 child=None, sibling=None):
        super().__init__(DAPICallSingle.name(), child, sibling)
        self.type = self.val


    def get_start_of_fp_nodes(self):
        return self.child.sibling

    def get_api_call_name(self):
        return self.child.val

    @staticmethod
    def name():
        return 'DAPICallSingle'

    def get_api_name(self):
        from program_helper.ast.ops import DAPIInvoke
        full_val = self.child.val
        return DAPIInvoke.split_api_call(full_val)[0]

    @staticmethod
    def construct_blank_node():
        single_node = DAPICallSingle()
        _call = "System.out.println()"
        single_node.child = DAPICall(_call)
        single_node.child.sibling = DStop()
        return single_node
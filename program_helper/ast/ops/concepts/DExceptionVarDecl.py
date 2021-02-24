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

from program_helper.ast.ops import DVarDecl, DType


# This is indistinguishable from DVarCall, except being valid
class DExceptionVarDecl(DVarDecl):
    def __init__(self, node_js=None, symtab=None, child=None, sibling=None):
        super().__init__(node_js, symtab, child, sibling)
        # self.invalidate()  # by default is invalid
        self.type = DExceptionVarDecl.name()
        self.val = self.type

    @staticmethod
    def name():
        return 'DExceptionVar'

    @staticmethod
    def construct_blank_node():
        var_node = DExceptionVarDecl()
        var_node.child = DType("java.lang.Object")
        return var_node

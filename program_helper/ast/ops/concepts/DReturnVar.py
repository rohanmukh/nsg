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

from program_helper.ast.ops import Node, DVarAccess, DAPIInvoke
from program_helper.ast.parser.ast_exceptions import TypeMismatchException


class DReturnVar(Node):
    def __init__(self, node_js=None, symtab=None, child=None, sibling=None):
        super().__init__(DReturnVar.name(), child, sibling)

        if node_js is None:
            return

        self.var_id = node_js['_id'] if '_id' in node_js \
                                        and node_js['_id'] is not None \
                                        and node_js['_id'] != "null" \
            else "no_return_var"

        self.var_type = node_js['_returns'] if '_returns' in node_js else None

        if self.var_type != symtab['return_type']:
            raise TypeMismatchException

        self.var_id = DAPIInvoke.validate_and_get_id(self.var_id, symtab)
        assert self.var_id != 'LITERAL'
        self.child = DVarAccess(self.var_id, return_type=self.var_type)
        self.iattrib = node_js["iattrib"] if 'iattrib' in node_js else [0, 0, 0]
        self.type = self.val

    @staticmethod
    def name():
        return 'DReturnVar'

    def get_ret_id(self):
        return self.child.val

    @staticmethod
    def construct_blank_node():
        var_node = DReturnVar()
        var_node.child = 0
        return var_node

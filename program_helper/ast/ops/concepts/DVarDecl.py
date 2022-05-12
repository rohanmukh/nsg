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

from program_helper.ast.ops import Node, DType, DSymtabMod, DVarAccess

class DVarDecl(Node):
    def __init__(self, node_js=None, symtab=None, child=None, sibling=None):
        super().__init__(DVarDecl.name(), child, sibling)

        self.type = self.val

        if node_js is None:
            return

        self.var_id = str(node_js['_id']) if '_id' in node_js else None
        self.var_type = node_js['_returns'] if '_returns' in node_js else None

        # self.child = DVarAccess(self.var_id)
        self.child = DType(self.var_type)
        self.child.sibling = DSymtabMod(0, type_helper=self.var_type)

        self.invalidate()  # by default is invalid

        self.iattrib = node_js["iattrib"] if 'iattrib' in node_js else [0, 0, 0]
        # Update symtab:
        self.update_symtab(symtab)

    @staticmethod
    def name():
        return 'DVarDecl'

    def update_symtab(self, symtab):
        symtab[self.var_id] = self

    def invalidate(self):
        self.valid = False
        temp = self.child
        while temp is not None:
            temp.valid = False
            temp = temp.sibling

    def make_valid(self):
        self.valid = True
        temp = self.child
        while temp is not None:
            temp.valid = True
            temp = temp.sibling

    def get_var_id(self):
        return self.var_id

    # def set_var_id(self, val):
    #     self.var_id = val
    #     return

    def get_return_type(self):
        return self.child.val

    # def get_synthesized_id(self):
    #     return self.child.val

    @staticmethod
    def construct_blank_node():
        var_node = DVarDecl()
        var_node.child = "java.lang.Object"
        return var_node

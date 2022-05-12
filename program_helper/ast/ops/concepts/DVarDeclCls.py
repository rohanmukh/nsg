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
from program_helper.ast.ops import DVarAccess
from program_helper.ast.ops.concepts.DVarDecl import DVarDecl
from program_helper.ast.ops.leaf_ops.DClsType import DClsType
from program_helper.ast.ops.leaf_ops.DSymtabMod import DSymtabMod


class DVarDeclCls(DVarDecl):
    def __init__(self, node_js=None, symtab=None, child=None, sibling=None):
        super().__init__(node_js=node_js, symtab=symtab, child=child, sibling=sibling)

        self.val = DVarDeclCls.name()
        self.type = self.val

        if node_js is None:
            return

        self.var_id = str(node_js['_id']) if '_id' in node_js else None
        self.var_type = node_js['_returns'] if '_returns' in node_js else None

        self.child = DClsType(self.var_type)
        self.child.sibling = DSymtabMod(0, type_helper=self.var_type)

        self.invalidate()  # by default is invalid


        self.iattrib = node_js["iattrib"] if 'iattrib' in node_js else [0, 0, 0]

        # Update symtab:
        self.update_symtab(symtab)


    @staticmethod
    def name():
        return 'DVarDeclCls'

    @staticmethod
    def construct_blank_node():
        var_node = DVarDeclCls()
        var_node.child = "java.lang.Object"
        return var_node

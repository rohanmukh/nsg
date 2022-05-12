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

from program_helper.ast.ops import Node, DVarAccess, DAPIInvoke, DAPICallMulti
from program_helper.ast.ops.leaf_ops.DInternalMethodAccess import DInternalMethodAccess


class DInternalAPICall(Node):
    def __init__(self, node_js=None, symtab=None, child=None, sibling=None):
        super().__init__(DInternalAPICall.name(), child, sibling)

        self.type = self.val

        if node_js is None:
            return

        # add leaf node of Call
        self.var_type = node_js['_returns']
        self.var_id = node_js['ret_var_id'] if 'ret_var_id' in node_js \
                                               and node_js['ret_var_id'] is not None \
            else "no_return_var"

        self.var_id = DAPIInvoke.validate_and_get_id(self.var_id, symtab)

        mod_arg_ids = DAPIInvoke.get_arg_ids(node_js["fp_var_ids"], symtab)
        arg_types = node_js['int_meth_formal_types']
        int_meth_id = node_js['int_meth_id']
        self.child = DInternalMethodAccess(int_meth_id, return_type=self.var_type)

        self.child.sibling = DVarAccess(self.var_id,
                                        return_type=self.var_type)
        self.child.sibling.sibling = self.get_fp_nodes(arg_types, mod_arg_ids)

        self.iattrib = node_js["iattrib"] if 'iattrib' in node_js else [0, 0, 0]

    def get_fp_nodes(self, arg_types, arg_ids):
        fp_head = fp_node = Node()
        for i, type in enumerate(arg_types):
            if i < len(arg_ids):
                fp_node.sibling = DVarAccess(arg_ids[i], type_helper=type)
                fp_node = fp_node.sibling
        return fp_head.sibling


    @staticmethod
    def name():
        return 'DInternalAPICall'

    def get_return_id(self):
        return self.child.val

    def get_internal_api_id(self):
        return self.child.val

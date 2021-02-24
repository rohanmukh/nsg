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

from program_helper.ast.ops import DAPICall, DAPIInvoke, Node, DVarAccessDecl, DSymtabMod
from program_helper.ast.ops.concepts.DAPICallMulti import DAPICallMulti
from program_helper.ast.ops.concepts.DAPICallSingle import DAPICallSingle
from program_helper.ast.ops.leaf_ops.DClsType import DClsType
from program_helper.ast.parser.ast_exceptions import UnknownVarAccessException
from utilities.vocab_building_dictionary import DELIM


class DClsInit(Node):
    def __init__(self, node_js=None, symtab=None, child=None, sibling=None):
        super().__init__(DClsInit.name(), child, sibling)

        if node_js is None:
            return

        self.expr_id = DELIM

        # if '_id' in node_js and '_type' in node_js and node_js['_id'] is not None:

        if '_id' in node_js \
                and node_js['_id'] is not None:
            self.var_id = node_js['_id']
        elif 'ret_var_id' in node_js \
                and node_js['ret_var_id'] is not None:
            self.var_id = node_js['ret_var_id']
        else:
            self.var_id = "no_return_var"

        self.var_type = node_js['_returns']
        expr_type = DELIM

        if self.var_type is not None:
            # assert ret_type == self.var_type
            _call = node_js['_call'] + self.delimiter() + expr_type + self.delimiter() + self.var_type
            mod_arg_ids = DAPIInvoke.get_arg_ids(node_js["fp"], symtab)

            self.child = DVarAccessDecl(self.var_id)
            self.child.sibling = DClsType(self.var_type)

            self.child.sibling.sibling = DAPICallSingle(
                child=DAPICall(_call, type_helper=self.var_type,
                               sibling=DAPICallMulti.get_fp_nodes(_call, mod_arg_ids)
                               ))

            self.child.sibling.sibling.sibling = DSymtabMod(0, type_helper=self.var_type)

            # symtab[self.var_id] = self
            self.invalidate()  # by default is invalid

            self.update_symtab(symtab)
        else:
            self.valid = False

        self.iattrib = node_js["iattrib"] if 'iattrib' in node_js else [0, 0, 0]

        self.type = self.val

    def update_symtab(self, symtab):
        symtab[self.var_id] = self

    def get_child(self):
        return self.child

    @staticmethod
    def name():
        return 'DClsInit'

    @staticmethod
    def delimiter():
        return DAPIInvoke.delimiter()

    def get_return_type(self):
        return self.var_type

    def get_return_id(self):
        return self.child.sibling.val

    #
    # def update_var_id(self, symtab):
    #     symtab[self.var_id] = self
    #

    def invalidate(self):
        self.valid = False
        self.child.valid = False
        self.child.sibling.valid = False
        self.child.sibling.sibling.valid = False
        self.child.sibling.sibling.sibling.valid = False
        temp = self.child.sibling.sibling.child
        while temp is not None:
            temp.valid = False
            temp = temp.sibling

    def get_type(self, id, symtab):
        assert id not in ['system_package', 'LITERAL']

        if id not in symtab:
            return None
            # raise UnknownVarAccessException
        else:
            var_node = symtab[id]
            # symtab[self.var_id] = self
            return var_node.var_type

    def make_valid(self):
        self.valid = True
        self.child.valid = True
        self.child.sibling.valid = True
        self.child.sibling.sibling.valid = True
        self.child.sibling.sibling.sibling.valid = True
        temp = self.child.sibling.sibling.child
        while temp is not None:
            temp.valid = True
            temp = temp.sibling

    @staticmethod
    def construct_blank_node():
        cls_node = DClsInit()
        cls_node.child = DAPICallSingle.construct_blank_node()
        cls_node.child.sibling = DVarAccessDecl(val="no_return_var")
        return cls_node

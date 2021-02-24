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

from program_helper.ast.ops import Node, DVarAccess
from program_helper.ast.ops.concepts.DAPICallMulti import DAPICallMulti
from program_helper.ast.parser.ast_exceptions import UnknownVarAccessException


class DAPIInvoke(Node):
    def __init__(self, node_js=None, symtab=None, child=None, sibling=None):
        super().__init__(DAPIInvoke.name(), child, sibling)

        self.type = self.val

        if node_js is None:
            return

        # add leaf node of Call
        self.expr_id = node_js['expr_var_id'] if 'expr_var_id' in node_js and node_js['expr_var_id'] \
                                                 is not None else "system_package"
        self.var_type = node_js['_returns']
        self.var_id = node_js['ret_var_id'] if 'ret_var_id' in node_js \
                                               and node_js['ret_var_id'] is not None \
            else "no_return_var"

        first_node = node_js['_calls'][0]
        def_expr_type = DAPIInvoke.get_expr_type_from_key(first_node['_call'])
        self.expr_id = DAPIInvoke.validate_and_get_id(self.expr_id, symtab)
        self.var_id = DAPIInvoke.validate_and_get_id(self.var_id, symtab)
        _call = first_node['_call'] + self.delimiter() + def_expr_type + self.delimiter() + self.var_type

        mod_arg_ids = DAPIInvoke.get_arg_ids(first_node["fp"], symtab)
        self.child = DAPICallMulti(_call, arg_ids=mod_arg_ids)

        self.child.sibling = DVarAccess(self.expr_id,
                                        expr_type=def_expr_type)  # This is the return var
        self.child.sibling.sibling = DVarAccess(self.var_id,
                                                return_type=self.var_type)

        self.iattrib = node_js["iattrib"] if 'iattrib' in node_js else [0, 0, 0]
        # TODO: Can also support DAPISingle separately
        next_nodes = node_js['_calls'][1:]
        if len(next_nodes) > 0:
            for node in next_nodes:
                self.add_more_call(node, symtab)



    def get_first_call_node(self):
        return self.child.get_first_child()

    @staticmethod
    def get_arg_ids(fp_var_ids, symtab):
        mod_arg_ids = []
        for arg_id in fp_var_ids:
            if arg_id is None:
                arg_id = 'LITERAL'
            mod_id = DAPIInvoke.validate_and_get_id(arg_id, symtab)
            mod_arg_ids.append(mod_id)
        return mod_arg_ids

    @staticmethod
    def validate_and_get_id(id, symtab):
        assert id is not None
        if id in ['system_package', 'LITERAL', 'no_return_var']:
            return id
        elif id not in symtab:
            raise UnknownVarAccessException
        else:
            var_node = symtab[id]
            var_node.make_valid()
            return id

    def add_more_call(self, node_js, symtab):
        def_expr_type = DAPIInvoke.get_expr_type_from_key(node_js['_call'])
        _call = node_js['_call'] + self.delimiter() + def_expr_type + self.delimiter() + node_js['_returns']

        mod_arg_ids = DAPIInvoke.get_arg_ids(node_js["fp"], symtab)
        self.child.add_more_call(_call, arg_ids=mod_arg_ids)

        # Update the return type
        self.var_type = node_js['_returns']
        self.child.sibling.sibling.return_type = self.var_type
        return

    @staticmethod
    def name():
        return 'DAPIInvoke'

    @staticmethod
    def delimiter():
        return "__$$__"

    @staticmethod
    def split_api_call(call):
        '''
        :param call:
        :return: a triple of api name, expr_type and ret_type, in this order
        '''
        vals = call.split(DAPIInvoke.delimiter())
        assert len(vals) == 3
        return vals[0], vals[1], vals[2]

    # This expresson type is not useful, not collected from data
    @staticmethod
    def get_expr_type_from_key(key):
        expr_type = ".".join(key.split('(')[0].split('.')[:-1]).replace('$NOT$', '')
        return expr_type

    def get_expr_id(self):
        return self.child.sibling.val

    def get_return_id(self):
        return self.child.sibling.sibling.val

    def get_api_name(self):
        full_val = self.child.child.child.val
        return DAPIInvoke.split_api_call(full_val)[0]

    @staticmethod
    def construct_blank_node():
        invoke_node = DAPIInvoke()
        invoke_node.child = DAPICallMulti.construct_blank_node()
        invoke_node.child.sibling = DVarAccess(val="system_package")
        invoke_node.child.sibling.sibling = DVarAccess(val="no_return_var")
        return invoke_node

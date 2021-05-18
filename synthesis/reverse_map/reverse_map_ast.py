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
from program_helper.ast.ops import DAPIInvoke
from synthesis.ops.candidate_ast import SYMTAB_MOD, TYPE_NODE, API_NODE, VAR_NODE, OP_NODE, METHOD_NODE, CLSTYPE_NODE, \
    VAR_DECL_NODE


class AstReverseMapper:
    def __init__(self, vocab):
        self.vocab = vocab

        self.nodes, self.edges, self.targets = [], [], []
        self.var_decl_ids = []
        self.node_type_numbers = []
        self.type_helper_val, self.expr_type_val, self.ret_type_val = [], [], []
        self.num_data = 0
        self.gnn_node_infos = []
        if hasattr(vocab, 'gnn_node_dict') and vocab.gnn_node_dict is not None:
            self.ifgnn2nag = True
        else:
            self.ifgnn2nag = False
        return

    def add_data(self, nodes, edges, targets,
                 var_decl_ids,
                 node_type_number,
                 type_helper_val, expr_type_val, ret_type_val, gnn_info=None):

        self.nodes.extend(nodes)
        self.edges.extend(edges)
        self.targets.extend(targets)
        self.var_decl_ids.extend(var_decl_ids)

        self.node_type_numbers.extend(node_type_number)

        self.type_helper_val.extend(type_helper_val)
        self.expr_type_val.extend(expr_type_val)
        self.ret_type_val.extend(ret_type_val)
        self.num_data += len(nodes)

        if gnn_info is not None:
            self.gnn_node_infos.extend(gnn_info)

    def get_element(self, id):
        if self.ifgnn2nag:
            return self.nodes[id], self.edges[id], self.targets[id], \
                self.var_decl_ids[id], \
                self.node_type_numbers[id], \
                self.type_helper_val[id], self.expr_type_val[id], self.ret_type_val[id], \
                self.gnn_node_infos[id]
        else:
            return self.nodes[id], self.edges[id], self.targets[id], \
                self.var_decl_ids[id], \
                self.node_type_numbers[id], \
                self.type_helper_val[id], self.expr_type_val[id], self.ret_type_val[id]

    def decode_ast_paths(self, ast_element, partial=True):

        if self.ifgnn2nag:
            nodes, edges, targets, \
            var_decl_ids, \
            node_type_numbers, \
            type_helper_vals, expr_type_vals, ret_type_vals, \
                gnn_info = ast_element
        else:
            nodes, edges, targets, \
            var_decl_ids, \
            node_type_numbers, \
            type_helper_vals, expr_type_vals, ret_type_vals = ast_element

        for node in nodes:
            print(self.vocab.chars_concept[node], end=',')
        print()
        #
        for edge in edges:
            print(edge, end=',')
        print()

        if self.ifgnn2nag:
            for gnn_node in gnn_info['node_labels']:
                print(gnn_node, end=',')
            print()
            iter_object = ast_element[:-1]
        else:
            iter_object = ast_element
        for _, _, target, \
            var_decl_id, \
            node_type_numbers, \
            type_helper_val, expr_type_val, ret_type_val in zip(*iter_object):
            if node_type_numbers == SYMTAB_MOD:
                print('--symtab--', end=',')
            elif node_type_numbers == VAR_NODE:
                print(self.vocab.chars_var[target], end=',')
            elif node_type_numbers == VAR_DECL_NODE:
                print(self.vocab.chars_var[target], end=',')
            elif node_type_numbers == TYPE_NODE:
                print(self.vocab.chars_type[target], end=',')
            elif node_type_numbers == CLSTYPE_NODE:
                print(self.vocab.chars_type[target], end=',')
            elif node_type_numbers == API_NODE:
                api = self.vocab.chars_api[target]
                api = api.split(DAPIInvoke.delimiter())[0]
                print(api, end=',')
            elif node_type_numbers == OP_NODE:
                op = self.vocab.chars_op[target]
                print(op, end=',')
            elif node_type_numbers == METHOD_NODE:
                op = self.vocab.chars_method[target]
                print(op, end=',')
            else:
                print(self.vocab.chars_concept[target], end=',')
        print()

        if not partial:
            for var_decl_id in var_decl_ids:
                print(var_decl_id, end=',')
            print()

            for type_helper_val in type_helper_vals:
                print(self.vocab.chars_type[type_helper_val], end=',')
            print()

            for expr_type_val in expr_type_vals:
                print(self.vocab.chars_type[expr_type_val], end=',')
            print()

            for ret_type_val in ret_type_vals:
                print(self.vocab.chars_type[ret_type_val], end=',')
            print()

            print()

    def reset(self):
        self.nodes, self.edges, self.targets = [], [], []
        self.var_decl_ids = []
        self.node_type_numbers = []
        self.type_helper_val, self.expr_type_val, self.ret_type_val = [], [], []
        self.num_data = 0

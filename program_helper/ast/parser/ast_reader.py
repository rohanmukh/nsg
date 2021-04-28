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

import numpy as np
import pickle

from program_helper.ast.ops import DType, DVarAccess, DAPICall, DSymtabMod, DClsInit, DVarAssign, DAPIInvoke, \
    DAPICallMulti, DVarAccessDecl
from program_helper.ast.ops.leaf_ops.DClsType import DClsType
from program_helper.ast.ops.leaf_ops.DInternalMethodAccess import DInternalMethodAccess
from program_helper.ast.ops.leaf_ops.DOp import DOp
from program_helper.ast.parser.ast_parser import AstParser
from program_helper.ast.parser.ast_checker import AstChecker
from program_helper.ast.parser.ast_traverser import AstTraverser
from synthesis.ops.candidate_ast import CONCEPT_NODE, VAR_NODE, API_NODE, TYPE_NODE, SYMTAB_MOD, OP_NODE, METHOD_NODE, \
    CLSTYPE_NODE, VAR_DECL_NODE


class AstReader:

    def __init__(self, max_depth=32,
                 max_loop_num=None,
                 max_branching_num=None,
                 max_variables=None,
                 max_trys=None,
                 concept_vocab=None,
                 api_vocab=None,
                 type_vocab=None,
                 var_vocab=None,
                 op_vocab=None,
                 method_vocab=None,
                 infer=True,
                 ifgnn2nag=False):

        self.ast_parser = AstParser()
        self.ast_checker = AstChecker(
            max_depth=max_depth,
            max_loop_num=max_loop_num,
            max_branching_num=max_branching_num,
            max_trys=max_trys,
            max_variables=max_variables)

        self.max_depth = max_depth
        self.max_variables = max_variables
        self.infer = infer

        self.concept_vocab = concept_vocab
        self.api_vocab = api_vocab
        self.type_vocab = type_vocab
        self.var_vocab = var_vocab
        self.op_vocab = op_vocab
        self.method_vocab = method_vocab

        self.nodes = None
        self.edges = None
        self.targets = None

        self.var_decl_ids = None
        self.return_reached = None

        self.node_type_number = None

        self.type_helper_val = None
        self.expr_type_val = None
        self.ret_type_val = None

        self.iattrib = None
        self.all_var_mappers = None

        # GNN2NAG inputs.
        self.ifgnn2nag = ifgnn2nag
        self.eg_schedule = None
        self.gnn2nag_edges = None
        self.eg_sending_node_ids = None
        self.eg_msg_target_node_ids = None
        self.eg_receiving_node_ids = None
        self.eg_receiving_node_nums = None
        return

    def read_while_vocabing(self, program_ast_js, symtab=None, fp_type_head=None, field_head=None, repair_mode=True):
        # Read the Program AST to a sequence
        if symtab is None:
            symtab = dict()

        self.ast_node_graph, all_var_mappers = self.ast_parser.get_ast_with_memory(program_ast_js['_nodes'], symtab,
                                                                  fp_type_head=fp_type_head,
                                                                  field_head=field_head,
                                                                  repair_mode=repair_mode)

        if repair_mode:
            self.ast_checker.check(self.ast_node_graph)

        path = AstTraverser.depth_first_search(self.ast_node_graph)
        if self.ifgnn2nag:
            path_with_edges= AstTraverser.dfs_travesal_with_edges(
                self.ast_node_graph)
            gnn_info = AstTraverser.brockschmidt_traversal(
                self.ast_node_graph,
                path_with_edges[0],
                path_with_edges[2],
                path_with_edges[3])

            #gnn_results = AstTraverser.calculate_gnn_info(eg_schedule)
            #gnn_info = {
            #    'edge_info': path_with_edges[1],
            #    'eg_sending_node_ids': gnn_results[0],
            #    'eg_msg_target_node_ids': gnn_results[1],
            #    'eg_receiving_node_ids': gnn_results[2],
            #    'eg_receiving_node_num': gnn_results[3]
            #}

        parsed_ast_array = []
        parent_call_val = 0
        for i, (curr_node_val, curr_node_type, curr_node_validity,
                curr_node_var_decl_ids, curr_node_return_reached,
                parent_node_id,
                edge_type, expr_type, type_helper, return_type,
                iattrib) in enumerate(path):

            assert curr_node_validity is True

            node_type_number = CONCEPT_NODE
            expr_type_val, type_helper_val, ret_type_val = 0, 0, 0

            if curr_node_type == DType.name():
                node_type_number = TYPE_NODE
                value = self.type_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
            elif curr_node_type == DClsType.name():
                node_type_number = CLSTYPE_NODE
                value = self.type_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
            elif curr_node_type == DVarAccess.name():
                node_type_number = VAR_NODE
                value = self.var_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
                type_helper_val = self.type_vocab.conditional_add_or_get_node_val(type_helper, self.infer)
                expr_type_val = self.type_vocab.conditional_add_or_get_node_val(expr_type, self.infer)
                ret_type_val = self.type_vocab.conditional_add_or_get_node_val(return_type, self.infer)
            elif curr_node_type == DVarAccessDecl.name():
                node_type_number = VAR_DECL_NODE
                value = self.var_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
            elif curr_node_type == DAPICall.name():
                node_type_number = API_NODE
                value = self.api_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)

                ## Even though the expr_type, ret_type are not part of data extracted now
                ## they can be when random apicalls are invoked during testing
                _, expr_type, ret_type = DAPIInvoke.split_api_call(curr_node_val)
                arg_list = DAPICallMulti.get_formal_types_from_data(curr_node_val)
                _ = self.type_vocab.conditional_add_or_get_node_val(expr_type, self.infer)
                _ = self.type_vocab.conditional_add_or_get_node_val(ret_type, self.infer)
                for arg in arg_list:
                    _ = self.type_vocab.conditional_add_or_get_node_val(arg, self.infer)

            elif curr_node_type == DSymtabMod.name():
                node_type_number = SYMTAB_MOD
                value = 0
                type_helper_val = self.type_vocab.conditional_add_or_get_node_val(type_helper, self.infer)
            elif curr_node_type == DOp.name():
                node_type_number = OP_NODE
                value = self.op_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
            elif curr_node_type == DInternalMethodAccess.name():
                node_type_number = METHOD_NODE
                value = self.method_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
            else:
                node_type_number = CONCEPT_NODE
                value = self.concept_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)

            # now parent id is already evaluated since this is top-down breadth_first_search
            parent_id = path[parent_node_id][0]
            parent_type = path[parent_node_id][1]
            if parent_type not in [DType.name(), DClsType.name(), DAPICall.name(), DVarAccess.name(),  DVarAccessDecl.name(), DSymtabMod.name(), DOp.name()]:
                parent_call_val = self.concept_vocab.get_node_val(parent_id)

            if value is not None and i > 0:
                parsed_ast_array.append((parent_call_val, edge_type, value,
                                         curr_node_var_decl_ids, curr_node_return_reached,
                                         node_type_number,
                                         type_helper_val, expr_type_val, ret_type_val,
                                         iattrib))

        if self.ifgnn2nag:
            return_items = (parsed_ast_array, all_var_mappers,
                            # TODO(ywen666): check which kinds of edges return
                            gnn_info)
            return return_items
        else:
            return parsed_ast_array, all_var_mappers

    # sz is total number of data points, Wrangle Program ASTs into numpy arrays
    def wrangle(self, ast_programs, all_var_mappers,
                min_num_data=None, gnn_info=None):
        if min_num_data is None:
            sz = len(ast_programs)
        else:
            sz = max(min_num_data, len(ast_programs))

        # TODO(ywen666): Hard-coding here!
        gnn_max_depth = 100
        self.nodes = np.zeros((sz, self.max_depth), dtype=np.int32)
        self.edges = np.zeros((sz, self.max_depth), dtype=np.bool)
        self.targets = np.zeros((sz, self.max_depth), dtype=np.int32)

        self.var_decl_ids = np.zeros((sz, self.max_depth), dtype=np.int32)
        self.return_reached = np.zeros((sz, self.max_depth), dtype=np.bool)

        self.node_type_number = np.zeros((sz, self.max_depth), dtype=np.int32)

        self.type_helper_val = np.zeros((sz, self.max_depth), dtype=np.int32)
        self.expr_type_val = np.zeros((sz, self.max_depth), dtype=np.int32)
        self.ret_type_val = np.zeros((sz, self.max_depth), dtype=np.int32)

        self.all_var_mappers = np.zeros((sz, self.max_variables*3), dtype=np.int32)
        self.iattrib = np.zeros((sz, self.max_depth, 3), dtype=np.bool)

        for i, api_path in enumerate(ast_programs):
            len_path = min(len(api_path), self.max_depth)
            mod_path = api_path[:len_path]
            self.nodes[i, :len_path] = [p[0] for p in mod_path]
            self.edges[i, :len_path] = [p[1] for p in mod_path]
            self.targets[i, :len_path] = [p[2] for p in mod_path]

            self.var_decl_ids[i, :len_path] = [p[3] for p in mod_path]
            self.return_reached[i, :len_path] = [p[4] for p in mod_path]

            self.node_type_number[i, :len_path] = [p[5] for p in mod_path]

            self.type_helper_val[i, :len_path] = [p[6] for p in mod_path]
            self.expr_type_val[i, :len_path] = [p[7] for p in mod_path]
            self.ret_type_val[i, :len_path] = [p[8] for p in mod_path]

            self.iattrib[i, :len_path, :] = [p[9] for p in mod_path]

            self.all_var_mappers[i] = all_var_mappers[i]

        if self.ifgnn2nag:
            self.gnn_info = gnn_info
        return

    def save(self, path):
        with open(path + '/ast_apis.pickle', 'wb') as f:
            pickle.dump([self.nodes, self.edges, self.targets,
                         self.var_decl_ids, self.return_reached,
                         self.node_type_number,
                         self.type_helper_val, self.expr_type_val,
                         self.ret_type_val,  self.all_var_mappers,
                         self.iattrib
                         ], f)
        if self.ifgnn2nag:
            with open(path + '/gnn2nag.pickle', 'wb') as f:
                pickle.dump(self.gnn_info, f)

    def load_data(self, path):
        with open(path + '/ast_apis.pickle', 'rb') as f:
            [self.nodes, self.edges, self.targets,
             self.var_decl_ids, self.return_reached,
             self.node_type_number,
             self.type_helper_val, self.expr_type_val,
             self.ret_type_val, self.all_var_mappers,
             self.iattrib
             ] = pickle.load(f)

        with open(path + '/gnn2nag.pickle', 'rb') as f:
            self.gnn_info = pickle.load(f)
        return

    def truncate(self, sz):
        self.nodes = self.nodes[:sz, :self.max_depth]
        self.edges = self.edges[:sz, :self.max_depth]
        self.targets = self.targets[:sz, :self.max_depth]

        self.var_decl_ids = self.var_decl_ids[:sz, :self.max_depth]
        self.return_reached = self.return_reached[:sz, :self.max_depth]

        self.node_type_number = self.node_type_number[:sz, :self.max_depth]

        self.type_helper_val = self.type_helper_val[:sz, :self.max_depth]
        self.expr_type_val = self.expr_type_val[:sz, :self.max_depth]
        self.ret_type_val = self.ret_type_val[:sz, :self.max_depth]

        self.all_var_mappers = self.all_var_mappers[:sz]
        self.iattrib = self.iattrib[:sz]
        return

    def split(self, num_batches, batch_size=128):
        # split into batches
        self.nodes = np.split(self.nodes, num_batches, axis=0)
        self.edges = np.split(self.edges, num_batches, axis=0)
        self.targets = np.split(self.targets, num_batches, axis=0)

        self.var_decl_ids = np.split(self.var_decl_ids, num_batches, axis=0)
        self.return_reached = np.split(self.return_reached, num_batches, axis=0)

        self.node_type_number = np.split(self.node_type_number, num_batches, axis=0)

        self.type_helper_val = np.split(self.type_helper_val, num_batches, axis=0)
        self.expr_type_val = np.split(self.expr_type_val, num_batches, axis=0)
        self.ret_type_val = np.split(self.ret_type_val, num_batches, axis=0)

        self.all_var_mappers = np.split(np.array(self.all_var_mappers), num_batches, axis=0)
        self.iattrib = np.split(np.array(self.iattrib), num_batches, axis=0)

        if self.ifgnn2nag:
            self.gnn_info = [self.gnn_info[i:i + batch_size]
                             for i in range(0, len(self.gnn_info), batch_size)]
        return

    def get(self):
        return self.nodes, self.edges, self.targets, \
               self.var_decl_ids, self.return_reached,\
               self.node_type_number, \
               self.type_helper_val, self.expr_type_val, self.ret_type_val, \
               self.all_var_mappers, self.iattrib

    def get_gnn_info(self):
        return self.gnn_info

    def add_data_from_another_reader(self, ast_reader):
        if self.nodes is None:
            assert self.edges is None and self.targets is None
            assert self.var_decl_ids is None and self.return_reached is None
            assert self.node_type_number is None and self.type_helper_val is None
            assert self.expr_type_val is None and self.ret_type_val is None
            assert self.iattrib is None and self.all_var_mappers is None

            self.nodes = ast_reader.nodes
            self.edges = ast_reader.edges
            self.targets = ast_reader.targets

            self.var_decl_ids = ast_reader.var_decl_ids
            self.return_reached = ast_reader.return_reached

            self.node_type_number = ast_reader.node_type_number

            self.type_helper_val = ast_reader.type_helper_val
            self.expr_type_val = ast_reader.expr_type_val
            self.ret_type_val = ast_reader.ret_type_val

            self.iattrib = ast_reader.iattrib
            self.all_var_mappers = ast_reader.all_var_mappers

        else:
            self.nodes = np.append(self.nodes, ast_reader.nodes, axis=0)
            self.edges = np.append(self.edges, ast_reader.edges, axis=0)
            self.targets = np.append(self.targets, ast_reader.targets, axis=0)

            self.var_decl_ids = np.append(self.var_decl_ids, ast_reader.var_decl_ids, axis=0)
            self.return_reached = np.append(self.return_reached, ast_reader.return_reached, axis=0)

            self.node_type_number = np.append(self.node_type_number, ast_reader.node_type_number, axis=0)

            self.type_helper_val = np.append(self.type_helper_val, ast_reader.type_helper_val, axis=0)
            self.expr_type_val = np.append(self.expr_type_val, ast_reader.expr_type_val, axis=0)
            self.ret_type_val = np.append(self.ret_type_val, ast_reader.ret_type_val, axis=0)

            self.iattrib = np.append(self.iattrib, ast_reader.iattrib, axis=0)
            self.all_var_mappers = np.append(self.all_var_mappers, ast_reader.all_var_mappers, axis=0)


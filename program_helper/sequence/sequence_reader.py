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

from program_helper.ast.ops import DVarDecl, DSubTree, DStop, DType, DSymtabMod, DVarAccess
from program_helper.ast.ops.concepts.DVarDeclCls import DVarDeclCls
from program_helper.ast.ops.leaf_ops.DClsType import DClsType
from program_helper.ast.ops.leaf_ops.DInternalMethodAccess import DInternalMethodAccess
from program_helper.ast.parser.ast_exceptions import TooManyVariableException
from program_helper.ast.parser.ast_traverser import AstTraverser


class SequenceReader:

    def __init__(self, max_depth=10,
                 type_vocab=None,
                 var_vocab=None,
                 concept_vocab=None,
                 infer=True):
        self.type_vocab = type_vocab
        self.var_vocab = var_vocab
        self.concept_vocab = concept_vocab

        self.max_depth = max_depth
        self.infer = infer

        self.fp_input = None
        return

    def form_fp_ast(self, js_formals, symtab):
        fp_head = curr_node = DSubTree()
        for fp_node in js_formals:
            type = fp_node['node']
            if type in [DVarDecl.name(), DVarDeclCls.name()]:
                if fp_node['_id'] is not None:
                    fp_node["iattrib"] = [0, 0, 0]
                    node = DVarDecl(fp_node, symtab)
                    curr_node = curr_node.add_and_progress_sibling_node(node)

        curr_node.add_sibling_node(DStop())
        fp_head.child = fp_head.sibling
        fp_head.sibling = None
        return fp_head


    def form_list(self, parsed_fp_array):
        fps = []
        for item in parsed_fp_array:
            parent_call_val, edge_type, value, type_or_not = item
            if type_or_not:
                fps.append(value)
        return fps


    def read_while_vocabing(self, fp_type_head):
        # Read the formal params
        path = AstTraverser.depth_first_search(fp_type_head)

        valid_fp_values = []
        parsed_fp_array = []
        parent_call_val = 0
        for i, (curr_node_val, curr_node_type, curr_node_validity,
                curr_node_var_decl_ids, curr_node_return_reached,
                parent_node_id,
                edge_type, expr_type, type_helper, return_type, _) in enumerate(path):

            assert curr_node_validity is True

            type_or_not = False
            if curr_node_type == DType.name():
                type_or_not = True
                value = self.type_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
                valid_fp_values.append(curr_node_val)
            elif curr_node_type == DClsType.name():
                type_or_not = True
                value = self.type_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)
                valid_fp_values.append(curr_node_val)
            elif curr_node_type == DSymtabMod.name():
                # We will skip the symtab updates for FPs
                continue
            elif curr_node_type in [DVarAccess.name(), DInternalMethodAccess.name()]:
                # We will skip the symtab updates for FPs
                continue
            else:
                value = self.concept_vocab.conditional_add_or_get_node_val(curr_node_val, self.infer)

            parent_id = path[parent_node_id][0]
            parent_type = path[parent_node_id][1]

            if parent_type not in [DType.name()]:
                parent_call_val = self.concept_vocab.get_node_val(parent_id)

            if value is not None and i > 0:
                parsed_fp_array.append((parent_call_val, edge_type, value, type_or_not))

        return parsed_fp_array, valid_fp_values

    # sz is total number of data points, Wrangle Formal Param Types
    def wrangle(self, formal_params, min_num_data=None):
        if min_num_data is None:
            sz = len(formal_params)
        else:
            sz = max(min_num_data, len(formal_params))

        fp_targets = np.zeros((sz, self.max_depth), dtype=np.int32)
        self.fp_input = np.zeros((sz, self.max_depth), dtype=np.int32)
        fp_type_or_not = np.zeros((sz, self.max_depth), dtype=np.bool)
        for i, fp_path in enumerate(formal_params):
            len_path = min(len(fp_path), self.max_depth)
            mod_path = fp_path[:len_path]
            fp_targets[i, :len_path] = [p[2] for p in mod_path]
            fp_type_or_not[i, :len_path] = [p[3] for p in mod_path]
            k = 0
            for t, b in zip(fp_targets[i, :len_path], fp_type_or_not[i, :len_path]):
                if b:
                    self.fp_input[i, k] = t
                    k += 1
        return

    def save(self, path):
        with open(path + '/formal_params.pickle', 'wb') as f:
            pickle.dump(self.fp_input, f)
        return

    def load_data(self, path):
        with open(path + '/formal_params.pickle', 'rb') as f:
            self.fp_input = pickle.load(f)
        return self.fp_input

    def truncate(self, sz):
        self.fp_input = self.fp_input[:sz, :self.max_depth]
        return

    def split(self, num_batches):
        self.fp_input = np.split(self.fp_input, num_batches, axis=0)
        return

    def get(self):
        return self.fp_input

    def add_data_from_another_reader(self, seq_reader):
        if self.fp_input is None:
            self.fp_input = seq_reader.fp_input
        else:
            self.fp_input = np.append(self.fp_input, seq_reader.fp_input, axis=0)


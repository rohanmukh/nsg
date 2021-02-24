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
from data_extraction.data_reader.data_reader import MAX_VARIABLES
from program_helper.ast.ops import DSubTree, CHILD_EDGE
import numpy as np


class CandidateElement:
    def __init__(self, curr_node=None,
                 edge_path=None,
                 curr_node_type=None,
                 curr_node_val=None,
                 next_node_type=None,
                 var_decl_id_val=None,
                 fixed_node_val=None,
                 return_reached=None
                 ):
        self.curr_node = curr_node
        self.edge_path = edge_path

        self.curr_node_val = curr_node_val
        self.curr_node_type = curr_node_type
        self.next_node_type = next_node_type
        self.var_decl_id_val = var_decl_id_val
        self.fixed_node_val = fixed_node_val
        self.return_reached = return_reached

    def get_current_node(self):
        return self.curr_node

    def get_edge_path(self):
        return self.edge_path

    def get_curr_node_type(self):
        return self.curr_node_type

    def get_curr_node_val(self):
        return self.curr_node_val

    def get_next_node_type(self):
        return self.next_node_type

    def get_next_var_decl_id_val(self):
        return self.var_decl_id_val

    def get_fixed_node_val(self):
        return self.fixed_node_val

    def get_return_reached(self):
        return self.return_reached

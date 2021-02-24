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
from program_helper.ast.ops.concepts.DAPICallSingle import DAPICallSingle
from program_helper.ast.ops.node import Node
from program_helper.ast.ops import DAPICall, DStop, DVarAccess
from program_helper.ast.parser.ast_exceptions import NestedAPIParsingException
import re


class DAPICallMulti(Node):
    def __init__(self, _call=None, arg_ids=None,
                 child=None, sibling=None):
        super().__init__(DAPICallMulti.name(), child, sibling)

        self.type = self.val

        if _call is None:
            return

        self.child = DAPICallSingle(
            child=DAPICall(_call,
                           sibling=DAPICallMulti.get_fp_nodes(_call, arg_ids)),
            sibling=DStop())

        self.num_single_calls = 1

    @staticmethod
    def name():
        return 'DAPICallMulti'

    def add_more_call(self, _call, arg_ids=None):
        if self.num_single_calls == 1:
            self.child.sibling = DAPICallSingle(
                child=DAPICall(_call,
                               sibling=self.get_fp_nodes(_call, arg_ids)),
                sibling=DStop())
        elif self.num_single_calls == 2:
            self.child.sibling.sibling = DAPICallSingle(
                child=DAPICall(_call,
                               sibling=self.get_fp_nodes(_call, arg_ids)),
                sibling=DStop())
        elif self.num_single_calls == 3:
            self.child.sibling.sibling.sibling = DAPICallSingle(
                child=DAPICall(_call,
                               sibling=self.get_fp_nodes(_call,
                                                         arg_ids)),
                sibling=DStop())
        else:
            raise NestedAPIParsingException
        self.num_single_calls += 1
        return

    def get_first_child(self):
        return self.child

    @staticmethod
    def get_fp_nodes(_call, arg_ids):
        arg_types = DAPICallMulti.get_formal_types_from_data(_call)
        fp_head = fp_node = Node()
        for i, type in enumerate(arg_types):
            if i < len(arg_ids):
                fp_node.sibling = DVarAccess(arg_ids[i], type_helper=type)
                fp_node = fp_node.sibling
        return fp_head.sibling

    # @staticmethod
    # def get_formal_types_from_data(key):
    #     args_str = re.findall('\([a-zA-Z0-9 ,<>_\[\]\.?@]*\)', key)[0][1:-1]
    #     fps = [item.strip() for item in args_str.split(',')] if len(args_str) > 0 else []
    #     return fps

    @staticmethod
    def get_formal_types_from_data(call):
        bracket = re.findall('\([a-zA-Z0-9 ,<>_\[\]\.?@]*\)', call)[0][1:-1]
        arg_list = []
        angle_count = 0
        curr_arg = ''
        for s in bracket:
            if s == '<':
                angle_count += 1
            elif s == '>':
                angle_count -= 1
            if s == ',' and angle_count == 0:
                if len(curr_arg) > 0:
                    arg_list.append(curr_arg)
                curr_arg = ''
            else:
                curr_arg += s

        if len(curr_arg) > 0:
            arg_list.append(curr_arg)

        return arg_list

    @staticmethod
    def construct_blank_node():
        multi_node = DAPICallMulti()
        multi_node.child = DAPICallSingle.construct_blank_node()
        multi_node.child.sibling = DStop()
        return multi_node

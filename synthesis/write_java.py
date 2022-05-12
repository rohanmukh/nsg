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

from __future__ import print_function

from copy import deepcopy

from program_helper.ast.ops import *
from program_helper.ast.ops.concepts.DAPICallMulti import DAPICallMulti
from program_helper.ast.ops.concepts.DExceptionVarDecl import DExceptionVarDecl
from program_helper.ast.ops.concepts.DInfix import DInfix
from program_helper.ast.ops.concepts.DInternalAPICall import DInternalAPICall
from program_helper.ast.ops.concepts.DReturnVar import DReturnVar


class Write_Java:
    def __init__(self, rename_vars=True):
        self.rename_vars = rename_vars
        return

    def add_comment(self, comment):
        comment_lines = comment.split('\n')
        header = ''
        for comment in comment_lines:
            header += "// " + comment + '\n'
        return header

    def program_synthesize_from_json(self, json_input,
                                     comment='',
                                     prob=None,
                                     beam_id=None,
                                     mapper=None):

        self.field_map_dict = {i: j for i, j in enumerate(mapper[20:])}
        self.fp_map_dict = {i: j for i, j in enumerate(mapper[10:20])}
        self.int_var_mapper_dict = {i: j for i, j in enumerate(mapper[:10])}

        prog = self.add_comment(comment)
        fts = json_input['field_types']
        for j, ft in enumerate(fts):
            if ft['_returns'] is not None:
                prog += ft['_returns'] + ' field_' + str( self.field_map_dict[j] )
                if j < len(fts) - 1:
                    prog += ', '
        prog += ';\n'

        if prob is not None:
            prog += "// log_prob :: " + str(prob) + '\n'
        if beam_id is not None:
            prog += "// beam id :: " + str(beam_id) + '\n'
        prog += json_input['return_type'] + " " + json_input["method"] + "("
        fps = json_input['formal_params']
        for j, fp in enumerate(fps):
            if fp['_returns'] is not None:
                prog += fp['_returns'] + ' fp_' + str( self.fp_map_dict[j] )
                if j < len(fps) - 1:
                    prog += ', '

        prog += '){\n'
        temp_prog , _= self.synthesize_body_from_json(json_input['ast']['_nodes'], tab_len=1, variable_count=0)
        prog += temp_prog
        prog += "\n}"
        return prog

    def handle_bracket(self, call, var_ids):
        if call == 'DAPICallSingle' or call == 'DAPICallMultiple'\
                or call == 'DStop' or call == 'DInfix' or '(' not in call:
            return call
        bracketed_str = '('
        arg_types = DAPICallMulti.get_formal_types_from_data(call)
        for i, (typ, fp) in enumerate(zip(arg_types, var_ids)):
            bracketed_str += typ + ': ' + fp
            bracketed_str += ',' if i < len(arg_types) - 1 else ''
        bracketed_str += ')'
        return bracketed_str

    def extract_apiname(self, call, expr_var):
        name = call.split('(')[0]
        name = name + '.' # The extra dot is a hack, makes sure last api is included
        name_comps = [] #name.split('.')
        word = ''
        stack = []
        for char in name:
            if char == '<':
                stack.append(1)
                word += char
            elif char == '>':
                stack.pop()
                word += char

            elif char == '.' and len(stack) == 0:
                new_word = deepcopy(word)
                name_comps.append(new_word)
                word = ''
            else:
                word += char

        if expr_var == "system_package":
            output = '.'.join(name_comps[-2:])
        else:
            output = name_comps[-1]
        return output

    def handle_DVarDecl(self, js, var_id, tab_len=0, clsinit_or_not=False):
        if self.rename_vars:
            if var_id < 10:
                id = "var_" + str(self.int_var_mapper_dict[var_id])
            else:
                id = "var_" + str(var_id)
        else:
            id = js['_id']
        # output_prog = "\t" * tab_len + "// Using stmt->DVarDecl " + "\n"
        output_prog = "\t" * tab_len + js['_returns'] + " " + id + ";\n"
        # output_prog += " // init an object \n" if clsinit_or_not else "\n"
        return output_prog

    def handle_DClsInit(self, js, tab_len=0):
        # output_prog = "\t" * tab_len + "// Using stmt->DClsInit " + "\n"
        output_prog = "\t" * tab_len
        output_prog += js["_id"] + " = " if js["_id"] != 'no_return_var' else ""

        call = js["_call"]
        output_prog += "new " + call.split('(')[0]
        output_prog += self.handle_bracket(call, js["fp"])
        output_prog += ";\n"
        return output_prog

    def handle_DInfix(self, js, tab_len=0, variable_count=0):
        output_prog = "\t" * tab_len
        output_prog += "\t" * tab_len
        temp_prog, _ = self.synthesize_body_from_json(js['_left'], tab_len=0, variable_count=variable_count)
        output_prog += temp_prog
        output_prog = output_prog[:-2] + " " + js["_op"][0] + " "
        temp_prog, _ = self.synthesize_body_from_json(js['_right'], tab_len=0, variable_count=variable_count)
        output_prog += temp_prog
        output_prog += ';\n'  # but this will be pruned
        return output_prog


    def handle_DAPIInvoke(self, js, tab_len=0):
        output_prog = "\t" * tab_len
        expr_var = js["expr_var_id"]
        output_prog += js["ret_var_id"] + " = " if js["ret_var_id"] != 'no_return_var' else ""
        output_prog += expr_var + "." if expr_var != "system_package" else ""
        for j, calls in enumerate(js['_calls']):
            call = calls["_call"]
            output_prog += self.extract_apiname(call, expr_var)
            output_prog += self.handle_bracket(call, calls['fp'])
            output_prog += "." if j < len(js['_calls']) - 1 else ";\n"
        return output_prog

    def handle_DBranch(self, js, tab_len=0, variable_count=0):
        output_prog = "\t" * tab_len + "if ("
        new_variable_count = variable_count
        if len(js['_cond']) == 0:
            output_prog += "true){\n"
        else:
            temp_prog, new_variable_count = self.synthesize_body_from_json(js['_cond'], tab_len=0, variable_count=variable_count)
            output_prog += temp_prog[:-2] + "){\n"

        temp_prog, _ = self.synthesize_body_from_json(js['_then'], tab_len=tab_len + 1, variable_count=new_variable_count)
        output_prog += temp_prog

        if len(js['_else']) > 0:
            output_prog += "\t" * tab_len + "}else {\n"
            temp_prog, _ = self.synthesize_body_from_json(js['_else'], tab_len=tab_len + 1, variable_count=new_variable_count)
            output_prog += temp_prog

        output_prog += "\t" * tab_len + "}\n"
        return output_prog

    def handle_DLoop(self, js, tab_len=0, variable_count=0):
        output_prog = "\t" * tab_len + "while ("
        new_variable_count = variable_count
        if len(js['_cond']) == 0:
            output_prog += "true){\n"
        else:
            temp_prog, new_variable_count = self.synthesize_body_from_json(js['_cond'], tab_len=0, variable_count=variable_count)
            output_prog += temp_prog[:-2] + "){\n"
        temp_prog, _ = self.synthesize_body_from_json(js['_body'], tab_len=tab_len + 1, variable_count=new_variable_count)
        output_prog += temp_prog
        output_prog += "\t" * tab_len + "}\n"
        return output_prog

    def handle_DExcept(self, js, tab_len=0, variable_count=0):
        output_prog = "\t" * tab_len + "try {\n"
        temp_prog, _ = self.synthesize_body_from_json(js['_try'], tab_len=tab_len + 1, variable_count=variable_count)
        output_prog += temp_prog
        output_prog += "\t" * tab_len + "}\n"
        output_prog += "\t" * tab_len + "catch("
        temp_prog, _ = self.synthesize_body_from_json(js['_catch'], tab_len=tab_len + 1, variable_count=variable_count)
        output_prog += temp_prog
        output_prog += "\t" * tab_len + ")\n"
        return output_prog

    def handle_DReturnVar(self, js, tab_len=0):
        output_prog = "\t" * tab_len
        _id = " " + js["_id"] if js["_id"] != 'no_return_var' else ""
        output_prog += "return" + _id + ";\n"
        return output_prog

    def handle_DStop(self, js, tab_len=0):
        output_prog = "\t" * tab_len + "}\n"
        return output_prog

    def handle_DSubTree(self, js, tab_len=0):
        output_prog = "\t" * tab_len + "{\n"
        temp_prog, _ = self.synthesize_body_from_json(js['_nodes'], tab_len=tab_len + 1)
        output_prog += temp_prog
        output_prog += "\t" * tab_len + "}\n"
        return output_prog

    def handle_DInternalAPICall(self, js, tab_len=0):
        output_prog = "\t" * tab_len
        output_prog += js["ret_var_id"] + " = " if js["ret_var_id"] != 'no_return_var' else ""
        output_prog += js["int_method_id"] + "("
        for i, fp in enumerate(js["fps"]):
            output_prog += fp
            output_prog += ',' if i < len(js["fps"]) - 1 else ''
        output_prog += ')'
        return output_prog


    def synthesize_body_from_json(self, json_array, tab_len=0, variable_count=0):
        output_prog = ""

        for js in json_array:

            if js["node"] == DSubTree.name():
                output_prog += self.handle_DSubTree(js, tab_len=tab_len)

            elif js["node"] in [DVarDecl.name(), DVarDeclCls.name(), DExceptionVarDecl.name()]:
                output_prog += self.handle_DVarDecl(js, tab_len=tab_len,
                                                    var_id=variable_count,
                                                    clsinit_or_not=DVarDeclCls.name() == js["node"]
                                                    )
                variable_count += 1

            elif js["node"] == DAPIInvoke.name():
                output_prog += self.handle_DAPIInvoke(js, tab_len=tab_len)

            elif js["node"] == DClsInit.name():
                output_prog += self.handle_DClsInit(js, tab_len=tab_len)

            # TODO: Does Infix even need variable count
            elif js["node"] == DInfix.name():
                output_prog += self.handle_DInfix(js, tab_len=tab_len, variable_count=variable_count)

            elif js["node"] == DBranch.name():
                output_prog += self.handle_DBranch(js, tab_len=tab_len, variable_count=variable_count)

            elif js["node"] == DLoop.name():
                output_prog += self.handle_DLoop(js, tab_len=tab_len, variable_count=variable_count)

            elif js["node"] == DExcept.name():
                output_prog += self.handle_DExcept(js, tab_len=tab_len, variable_count=variable_count)

            elif js["node"] == DStop.name():
                output_prog += self.handle_DStop(js, tab_len=tab_len)

            elif js["node"] == DReturnVar.name():
                output_prog += self.handle_DReturnVar(js, tab_len=tab_len)

            elif js["node"] == DInternalAPICall.name():
                output_prog += self.handle_DInternalAPICall(js, tab_len=tab_len)

            # elif js["node"] == DVarAssign.name():
            #     output_prog += "\t" * tab_len + js["_returns"] + " $" + js["_id"] + " = "
            #     output_prog += "$" + js["_rhs_id"] + ";\n"

            else:
                print("Unknown type " + js["node"] + " encountered " + "\n")
                # raise Exception

        return output_prog, variable_count



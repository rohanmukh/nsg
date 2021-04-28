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

import argparse
import sys

import ijson  # .backends.yajl2_cffi as ijson
import random
import os

from program_helper.ast.parser.ast_exceptions import TooLongBranchingException, TooLongLoopingException, \
    VoidProgramException, TooManyVariableException, UnknownVarAccessException, IgnoredForNowException, \
    TooManyTryException, NestedAPIParsingException, TooDeepException, TypeMismatchException, RetStmtNotExistException
from program_helper.program_reader import ProgramReader
from synthesis.write_java import Write_Java
from utilities.basics import conditional_director_creator
from utilities.logging import create_logger

MAX_AST_DEPTH = 64
MAX_FP_DEPTH = 10
MAX_FIELDS = 10
MAX_CAMELCASE = 3
MAX_KEYWORDS = 10
MAX_VARIABLES = 10

MAX_LOOP_NUM = 2
MAX_BRANCHING_NUM = 2
MAX_TRY_NUM = 2
SEED = 12


class Reader:
    def __init__(self,
                 dump_data_path=None,
                 infer=False,
                 infer_vocab_path=None,
                 repair_mode=True,
                 dump_ast=False,
                 ifgnn2nag=False
                 ):
        '''
        :param filename: JSON file to read from
        :param dump_data_path: data path to dump to
        :param infer_vocab_path: config to load from if infer
        :param infer: if used for inference
        '''

        assert infer_vocab_path is None if infer is False else not None

        self.infer = infer
        random.seed(SEED)
        conditional_director_creator(dump_data_path)
        self.logger = create_logger(os.path.join(dump_data_path, 'data_read.log'))
        self.logger.info('Reading data file...')
        self.ifgnn2nag = ifgnn2nag

        self.program_reader = ProgramReader(
            max_ast_depth=MAX_AST_DEPTH, max_loop_num=MAX_LOOP_NUM,
            max_branching_num=MAX_BRANCHING_NUM,
            max_fp_depth=MAX_FP_DEPTH,
            max_camel=MAX_CAMELCASE,
            max_fields=MAX_FIELDS,
            max_keywords=MAX_KEYWORDS,
            max_trys=MAX_TRY_NUM,
            max_variables=MAX_VARIABLES,
            data_path=dump_data_path,
            infer=infer,
            infer_vocab_path=infer_vocab_path,
            logger=self.logger,
            repair_mode=repair_mode,
            ifgnn2nag=ifgnn2nag
        )
        self.done, self.ignored_for_branch, self.ignored_for_loop, self.ignored_for_try, \
        self.ignored_for_illegal_var_access, self.ignored_for_nested_api, \
        self.ignored_for_void, self.ignored_for_more_vars, self.ignored_for_now, \
        self.ignored_for_depth, self.ignored_for_type, self.ignored_for_ret_stmt_not_exist, self.ignored_for_unknown\
            = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        self.data_points = None
        self.dump_ast = dump_ast

    def read_file(self, filename=None, prog_array=None, max_num_data=None):
        if filename is not None and prog_array is not None:
            raise Exception
        if filename is not None:
            prog_array_in = ijson.items(open(filename, 'rb'), 'programs.item')
        elif prog_array is not None:
            prog_array_in = prog_array
        else:
            raise Exception

        temp_data_points = self.read_data_from_array(prog_array_in, max_num_data=max_num_data)
        if self.data_points is None:
            self.data_points = temp_data_points
        else:
            self.data_points.extend(temp_data_points)

    def wrangle(self, min_num_data=0):

        # randomly shuffle to avoid bias towards initial data points during training
        random.shuffle(self.data_points)

        ##
        if self.ifgnn2nag:
            ast_programs, all_var_mappers, return_types, formal_params, \
                field_array, apicalls, types, keywords, \
                method, classname, javadoc_kws, surr_ret, \
                surr_fp, surr_method, surr_method_ids, \
                self.passed_jsons, self.real_javas, \
                self.checker_outcome_strings, self.asts, gnn_info  = \
                zip(*self.data_points)  # unzip
        else:
            ast_programs, all_var_mappers, return_types, formal_params, \
                field_array, apicalls, types, keywords, \
                method, classname, javadoc_kws, surr_ret, \
                surr_fp, surr_method, surr_method_ids, \
                self.passed_jsons, self.real_javas, \
                self.checker_outcome_strings, self.asts = \
                zip(*self.data_points)  # unzip

        # most programs share the same graph structures
        # Print the edges here
        #import pdb; pdb.set_trace()
        #test = gnn_info[0][0]
        #for item in gnn_info:
        #    print(test == item[0])
        #import pdb; pdb.set_trace()

        if self.passed_jsons[0] == None:
            self.passed_jsons = None
        if self.real_javas[0] == None:
            self.real_javas = None
        if self.checker_outcome_strings[0] == None:
            self.checker_outcome_strings = None
        if self.asts[0] == None:
            self.asts = None

        if self.ifgnn2nag:
            self.program_reader.wrangle(ast_programs, all_var_mappers,
                                        return_types, formal_params,
                                        field_array, apicalls, types,
                                        keywords, method, classname,
                                        javadoc_kws, surr_ret, surr_fp,
                                        surr_method, surr_method_ids,
                                        min_num_data=min_num_data,
                                        gnn_info = gnn_info)
        else:
            self.program_reader.wrangle(ast_programs, all_var_mappers,
                                        return_types, formal_params,
                                        field_array, apicalls, types,
                                        keywords, method, classname,
                                        javadoc_kws, surr_ret, surr_fp,
                                        surr_method, surr_method_ids,
                                        min_num_data=min_num_data)

    def dump(self):
        java_synthesis, javas_synthesized = Write_Java(rename_vars=True), list()
        program_jsons = list()
        if self.passed_jsons is not None:
            for json, outcome in zip(self.passed_jsons, self.checker_outcome_strings):
                try:
                    java_program = java_synthesis.program_synthesize_from_json(json, comment=outcome)
                except:
                    java_program = "Could not synthesize java program"
                javas_synthesized.append(java_program)
                program_jsons.append(json)

        self.program_reader.save_data(effective_javas=javas_synthesized,
                                      real_javas=self.real_javas,
                                      program_jsons=program_jsons,
                                      program_asts=self.asts)
        self.logger.info('Done!')

    def read_data_from_array(self, prog_array, max_num_data=None):
        data_points = []
        for program in prog_array:
            data_point = self.read_one_json_program(program)
            if data_point is None:
                continue

            data_points.append(data_point)
            self.done += 1
            if self.done % 1000 == 0 and self.done > 0:
                self.logger.info('Extracted data for {} programs'.format(self.done))
            if max_num_data is not None and self.done >= max_num_data:
                break
        self.logger.info('Extracted data for {} programs'.format(self.done))
        return data_points

    def read_one_json_program(self, program):
        data_point = None
        if 'ast' not in program:
            return None
        try:
            if self.ifgnn2nag:
                parsed_ast, all_var_mappers, return_type_id, \
                    parsed_fp_array, parsed_field_array, \
                    apicalls, types, keywords, \
                    method, classname, javadoc_kws, \
                    surr_ret, surr_fp, surr_method_names, surr_method_ids,\
                    mod_program_js, checker_outcome_string, gnn_info = \
                    self.program_reader.read_json(program)
            else:
                parsed_ast, all_var_mappers, return_type_id, \
                    parsed_fp_array, parsed_field_array, \
                    apicalls, types, keywords, \
                    method, classname, javadoc_kws, \
                    surr_ret, surr_fp, surr_method_names, surr_method_ids,\
                    mod_program_js, checker_outcome_string = \
                    self.program_reader.read_json(program)

            program_ast = program['ast'] if self.dump_ast or self.infer else None
            body = program['body'] if self.infer else None

            if self.ifgnn2nag:
                data_point = (
                    parsed_ast, all_var_mappers, return_type_id, \
                    parsed_fp_array, parsed_field_array, \
                    apicalls, types, keywords, method, classname, \
                    javadoc_kws, surr_ret, surr_fp, surr_method_names, \
                    surr_method_ids, mod_program_js, body, \
                    checker_outcome_string, program_ast, gnn_info)
            else:
                data_point = (
                    parsed_ast, all_var_mappers, return_type_id, \
                    parsed_fp_array, parsed_field_array, \
                    apicalls, types, keywords, method, classname, \
                    javadoc_kws, surr_ret, surr_fp, surr_method_names, \
                    surr_method_ids, mod_program_js, body, \
                    checker_outcome_string, program_ast)

        except TooLongLoopingException as e1:
            self.ignored_for_loop += 1

        except TooLongBranchingException as e2:
            self.ignored_for_branch += 1

        except TooManyTryException as e2:
            self.ignored_for_try += 1

        except NestedAPIParsingException as e2:
            self.ignored_for_nested_api += 1

        except UnknownVarAccessException as e3:
            self.ignored_for_illegal_var_access += 1

        except RetStmtNotExistException as e3:
            self.ignored_for_ret_stmt_not_exist += 1

        except VoidProgramException as e4:
            self.ignored_for_void += 1

        except TooManyVariableException as e5:
            self.ignored_for_more_vars += 1

        except IgnoredForNowException as e6:
            self.ignored_for_now += 1

        except TooDeepException as e6:
            self.ignored_for_depth += 1

        except TypeMismatchException as e7:
            self.ignored_for_type += 1

        return data_point

    def log_info(self):
        self.logger.info('{:8d} programs/asts in training data'.format(self.done))
        self.logger.info('{:8d} programs/asts missed in training data for loop'.format(self.ignored_for_loop))
        self.logger.info('{:8d} programs/asts missed in training data for branch'.format(self.ignored_for_branch))
        self.logger.info('{:8d} programs/asts missed in training data for try'.format(self.ignored_for_try))
        self.logger.info(
            '{:8d} programs/asts missed in training data for illegal var access'.format(
                self.ignored_for_illegal_var_access))
        self.logger.info('{:8d} programs/asts missed in training data for being void'.format(self.ignored_for_void))
        self.logger.info(
            '{:8d} programs/asts missed in training data for too many variables'.format(self.ignored_for_more_vars))
        self.logger.info('{:8d} programs/asts missed in training data for ignored now'.format(self.ignored_for_now))
        self.logger.info('{:8d} programs/asts missed in training data for ignored for nested api parsing'.format(
            self.ignored_for_nested_api))
        self.logger.info(
            '{:8d} programs/asts missed in training data for ignored for depth'.format(self.ignored_for_depth))
        self.logger.info(
            '{:8d} programs/asts missed in training data for ignored for type mismatch'.format(self.ignored_for_type))
        self.logger.info(
            '{:8d} programs/asts missed in training data for ignored for ret stmt not exist'.format(self.ignored_for_ret_stmt_not_exist))
        self.logger.info(
            '{:8d} programs/asts missed in training data for ignored for check'.format(self.ignored_for_unknown))

        #if self.infer:
        self.program_reader.ast_reader.ast_checker.print_stats(logger=self.logger)


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file', type=str, nargs=1,
                        help='input data file')
    parser.add_argument('--python_recursion_limit', type=int, default=100000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--data_path', type=str, default='data',
                        help='data to be saved here')

    clargs_ = parser.parse_args()
    sys.setrecursionlimit(clargs_.python_recursion_limit)
    filename_ = clargs_.input_file[0]

    r = Reader(dump_data_path=clargs_.data_path)
    r.read_file(filename=filename_, max_num_data=None)
    r.wrangle()
    r.log_info()
    r.dump()

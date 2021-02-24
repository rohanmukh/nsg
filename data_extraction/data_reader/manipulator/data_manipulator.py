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
from collections import defaultdict
from copy import deepcopy
from itertools import chain

import ijson
import argparse
import sys
import numpy as np

from data_extraction.data_reader.manipulator.data_add_attribs import DataAttributeLabeler
from data_extraction.data_reader.manipulator.data_preprocessor import DataPreProcessor
from data_extraction.data_reader.manipulator.dataread_exceptions import NotUniqException, lastAPIException
from data_extraction.data_reader.utils import gather_calls, gather_extra_infos
from program_helper.set.apicalls import ApiCalls
from program_helper.set.keywords import Keywords
from program_helper.set.types import Types
from utilities.basics import conditional_director_creator, dump_json


class DataManipulator:

    def __init__(self, debug_print=True):
        self.preprocessor = DataPreProcessor()
        self.program_dict = defaultdict(list)
        self.labeler = DataAttributeLabeler(dump_sattrib=False)
        self.unq_keys = set()
        self.print = debug_print

    def read_data(self, input_file,
                  output_file=None,
                  repair_mode=True
                  ):

        f = open(input_file, 'rb')
        # During the first stop repair of return types
        self.first_pass(f, repair_mode=repair_mode)
        all_programs = self.get_surrounding_data_in_second_pass(
            repair_mode=repair_mode)

        if output_file is not None:
            if self.print:
                print('Dumping!')
            dump_json({"size": len(all_programs), "programs": all_programs}, output_file)

        return all_programs if output_file is None else None

    def first_pass(self, f, repair_mode=True):
        '''
        :param f: a JSON file opened as read mode in bytes
        :return:
        '''
        if self.print:
            print('Starting First Pass')
        count, not_uniq, last_API, valid, unknown = 0, 0, 0, 0, 0
        for program in ijson.items(f, 'programs.item'):
            if 'ast' not in program:
                continue
            try:
                program = self.labeler.add_attributes(program)
                program = self.preprocess_program(program, repair_mode=repair_mode)
                self.add_to_program_dict(program, repair_mode=repair_mode)
                valid += 1

            except NotUniqException:
                not_uniq += 1
                pass

            except lastAPIException:
                last_API += 1
                pass

            except Exception:
                unknown += 1
                pass

            count += 1
            if count % 10000 == 0 and self.print:
                print('Number of programs processed : {}'.format(count))

        if self.print:
            print('Programs Processed: {}, Passed: {}, Failed for no-unq: {}, Last API exception {}, Unknow {}'.format(
                count, valid, not_uniq, last_API, unknown))
        return

    def preprocess_program(self, program, repair_mode=True):

        symtab = dict()
        field_json = self.preprocessor.full_preprocess(program['field_ast']['_nodes'], symtab=symtab)
        program['field_ast']['_nodes'] = field_json

        formal_param = self.preprocessor.full_preprocess(program['formalParam'],
                                                                 symtab=symtab)
        program['formal_params'] = formal_param
        del program['formalParam']

        ast_json = self.preprocessor.full_preprocess(program['ast']['_nodes'],
                                                     repair_mode=repair_mode,
                                                     symtab=symtab
                                                     )

        program['ast']['_nodes'] = ast_json
        if repair_mode:
            program['return_type'] = symtab['my_return_var']
        else:
            program['return_type'] = program['returnType']

        del program['returnType']

        apicalls, types, keywords = self.extract_keywords(program)

        program['apicalls'] = apicalls
        program['types'] = types
        program['keywords'] = keywords

        return program

    def extract_keywords(self, program):
        calls = gather_calls(program['ast'])

        apicalls = list(set(chain.from_iterable([ApiCalls.from_call(call)
                                                 for call in calls])))

        types = list(set(chain.from_iterable([Types.from_call(call)
                                              for call in calls])))

        keywords = list(set(chain.from_iterable([Keywords.from_call(call)
                                                 for call in calls])))


        keywords += gather_extra_infos(program['ast'])
        return apicalls, types, keywords


    def add_to_program_dict(self, program, repair_mode=True):
        unq_key = program['file'] + '/' + program['method']
        if unq_key not in self.unq_keys:
            self.unq_keys.add(unq_key)
        else:
            raise NotUniqException

        if len(program['ast']['_nodes']) > 0 or not repair_mode:
            file_key = program['file']
            self.program_dict[file_key].append(program)

    def get_surrounding_data_in_second_pass(self, repair_mode=True):
        if self.print:
            print('Starting Second Pass')
        count = 0
        new_programs_train = []
        for key, all_progs in self.program_dict.items():
            for prog in all_progs:
                # take single method during inference only selects one
                # program.
                if repair_mode is False \
                        and '@synthesize' not in prog['body']:
                    continue
                prog_copy = deepcopy(prog)
                prog_copy['Surrounding_Methods'] = list()
                internal_method_dict = dict()

                ten_random_surr_prog_id = np.random.permutation(len(all_progs))[:10]

                j = 0
                for choice_id in ten_random_surr_prog_id:
                    surr = all_progs[choice_id]
                    
                    if surr['method'] == prog_copy['method']:
                        continue

                    formal_types = []
                    for item in surr['formal_params']:
                        typ = item['_returns']
                        formal_types.append(typ)

                    internal_method_dict[surr['method']] = {'id': 'local_method_' + str(j),
                                                            'formal_types': formal_types,
                                                            }

                    temp = {'return_type': surr['return_type'],
                            'formal_params': formal_types,
                            'surr_method': surr['method'],
                            'id': j
                            }
                    prog_copy['Surrounding_Methods'].append(temp)
                    j += 1

                prog_copy['ast']['_nodes'] = self.modify_internal_apicall(prog['ast']['_nodes'],
                                                                          method_dict=internal_method_dict)

                new_programs_train.append(prog_copy)
                count += 1
                if count % 10000 == 0 and self.print:
                    print('Number of programs processed : {}'.format(count))

        if self.print:
            print('Programs Processed: {}'.format(count))
        return new_programs_train



    def modify_internal_apicall(self, node_array, method_dict=None):
        mod_nodes = []
        for node in node_array:
            if node["node"] == "DInternalAPICall":
                if node["int_meth_name"] not in method_dict:
                     continue
                node["int_meth_id"] = method_dict[node["int_meth_name"]]['id']
                node["int_meth_formal_types"] = method_dict[node["int_meth_name"]]['formal_types']
                del node["int_meth_name"]
            elif node["node"] == "DBranch":
                node['_cond'] = self.modify_internal_apicall(node['_cond'], method_dict=method_dict)
                node['_then'] = self.modify_internal_apicall(node['_then'], method_dict=method_dict)
                node['_else'] = self.modify_internal_apicall(node['_else'], method_dict=method_dict)
            elif node["node"] == "DLoop":
                node['_cond'] = self.modify_internal_apicall(node['_cond'], method_dict=method_dict)
                node['_body'] = self.modify_internal_apicall(node['_body'], method_dict=method_dict)
            elif node["node"] == "DExcept":
                node['_try'] = self.modify_internal_apicall(node['_try'], method_dict=method_dict)
                node['_catch'] = self.modify_internal_apicall(node['_catch'], method_dict=method_dict)
            elif node["node"] == "DInfix":
                node['_left'] = self.modify_internal_apicall(node['_left'], method_dict=method_dict)
                node['_right'] = self.modify_internal_apicall(node['_right'], method_dict=method_dict)
            else:
                pass
            mod_nodes.append(node)
        return mod_nodes

    def reset(self):
        self.program_dict.clear()
        self.unq_keys.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--python_recursion_limit', type=int, default=100000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--input_filename', type=str)
    parser.add_argument('--output_filename', type=str)
    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)

    conditional_director_creator('/'.join(clargs.output_filename.split('/')[:-1]))
    d = DataManipulator()
    d.read_data(clargs.input_filename, clargs.output_filename)

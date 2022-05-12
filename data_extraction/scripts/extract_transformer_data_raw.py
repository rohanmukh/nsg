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
import re
from multiprocessing import Pool
from data_extraction.data_reader.manipulator.dataread_exceptions import NotUniqException
from utilities.basics import conditional_director_creator, dump_json
from nltk.tokenize import word_tokenize

class MergerManipulator:
    def __init__(self, input_file_list=None,
                 output_folder=None,
                num_chunks=32):
        # super().__init__(input_file_list, output_file)
        self.num_chunks = num_chunks
        self.output_folder = output_folder
        conditional_director_creator(self.output_folder)

        self.chunked_file_list = [(_id, files) for _id, files in enumerate(self.split(input_file_list, chunks=self.num_chunks))]
        pool = Pool(processes=num_chunks)
        pool.map(self.read_a_chunk, self.chunked_file_list)


    def split(self, input_file_list, chunks=None):
        with open(input_file_list, errors='ignore') as f:
            file_list = f.readlines()
        chunked_file_list = np.array_split(np.array(file_list), chunks)
        return chunked_file_list

    def read_a_chunk(self, chunk):
        chunk_id, chunk_files = chunk[0], chunk[1]
        self.read_one_chunked_file_list(chunk_files,  chunk_id=chunk_id)
        return

    def read_one_chunked_file_list(self, file_list, chunk_id=None):
        manipulator = DataManipulator(debug_print=False)
        bodys, contexts, j = [], [], 0

        for j, filename in enumerate(file_list):
            filename = filename[:-1]  # ignore '\n'
            body, context = manipulator.read_data(filename, output_file=chunk_id)
            bodys.extend(body)
            contexts.extend(context)
            manipulator.reset()
            if (j+1) % 10000 == 0:
                print('Chunk id: {}, Loaded file: {}'.format(chunk_id, j+1), end='\n')

        print('Chunk id: {}, Loaded file: {}'.format(chunk_id, j+1), end='\n')

        print('Dumping!')
        self.write_file(bodys, "data_transformer/body_data_" + str(chunk_id) + ".original_subtoken")
        self.write_file(contexts, "data_transformer/context_data_" + str(chunk_id) + ".original")

        return

    def write_file(self, my_list, file_name):
        with open(file_name, 'w') as f:
            for item in my_list:
                f.write("%s\n" % item)



class DataManipulator:

    def __init__(self, debug_print=True):
        self.program_dict = defaultdict(list)
        self.unq_keys = set()
        self.print = debug_print


    def read_data(self, input_file,
                  output_file=None,
                  repair_mode=True
                  ):

        try:
            f = open(input_file, 'rb')
        except:
            return [], []

        # During the first stop repair of return types
        self.first_pass(f, repair_mode=repair_mode)
        bodys, contexts = self.get_surrounding_data_in_second_pass(
            repair_mode=repair_mode)
        return bodys, contexts


    def first_pass(self, f, repair_mode=True):
        '''
        :param f: a JSON file opened as read mode in bytes
        :return:
        '''
        if self.print:
            print('Starting First Pass')
        valid = 0
        for program in ijson.items(f, 'programs.item'):
            if 'ast' not in program:
                continue
            self.add_to_program_dict(program, repair_mode=repair_mode)
            valid += 1

        return


    def add_to_program_dict(self, program, repair_mode=True):
        unq_key = program['file'] + '/' + program['method']

        if len(program['ast']['_nodes']) > 0:
            file_key = program['file']
            self.program_dict[file_key].append(program)

    def get_surrounding_data_in_second_pass(self, repair_mode=True):
        if self.print:
            print('Starting Second Pass')
        count = 0
        bodys, contexts = [], []
        for key, all_progs in self.program_dict.items():
            for prog in all_progs:
                # take single method during inference only selects one
                # program.

                prog_copy = deepcopy(prog)
                javadoc = prog_copy["javaDoc"] if prog_copy["javaDoc"] is not None else " "

                surr_headers = list()
                ten_random_surr_prog_id = np.random.permutation(len(all_progs))[:10]
                j = 0
                for choice_id in ten_random_surr_prog_id:
                    surr = all_progs[choice_id]

                    if surr['method'] == prog_copy['method']:
                        continue

                    surr_header = self.extract_header(surr["body"])
                    surr_headers.append(surr_header)
                    j += 1

                surr_header_as_string = " ".join(surr_headers)
                method_header = self.extract_header(prog_copy["body"])
                fields = self.extract_field_infos(prog_copy["field_ast"]["_nodes"])
                classname = prog_copy["className"]

                context = [classname, fields, surr_header_as_string, javadoc, method_header]
                context_string = " ".join(context)
                contexts.append(self.tokenize(context_string))

                body = self.stripJavaDoc(prog_copy["body"])
                bodys.append(self.tokenize(body))

                count += 1

        return bodys, contexts

    def stripJavaDoc(self, stringBody):
        temp = re.sub(r'/\*\*(.*?)\*\/', '', stringBody.replace('\n', ''))
        temp = ' '.join([word for word in temp.split() if not word.startswith('@')])
        # temp = temp.replace('private', 'public')
        return temp

    def extract_header(self, input):
        temp = self.stripJavaDoc(input)
        try:
            body = re.findall('\{.*\}',temp)[0]
        except:
            body = re.findall('\{.*',temp)[0]
        header = temp.replace(body, '')
        return header

    def extract_field_infos(self, input):
        vals = []
        for item in input:
            if type in ['DFieldCall', 'DVarDecl', 'DVarDeclCls']:
                val = item["_returns"]
                vals.append(val)
        return " ".join(vals)

    def tokenize(self, sentence):
        sentence = sentence.replace(".", " . ")  # Add an space before and after the comma
        sentence = sentence.replace("=", " = ")  # Add an space before and after the equals
        sentence = sentence.replace("= =", "==")  # but make sure == stays the same
        sentence = sentence.replace("& &", "&&")  # but make sure && stays the same
        sentence = sentence.replace("| |", "||")  # but make sure || stays the same
        sentence = sentence.replace("< =", "<=")  # but make sure == stays the same
        sentence = sentence.replace("> =", ">=")  # but make sure == stays the same
        tokens = word_tokenize(sentence)
        return ' '.join(tokens)

    def reset(self):
        self.program_dict.clear()
        self.unq_keys.clear()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_list', type=str, nargs=1,
                        help='file containing list of all JSON files')
    parser.add_argument('--python_recursion_limit', type=int, default=100000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='file to output merged data')
    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)
    MergerManipulator(input_file_list=clargs.file_list[0],
           output_folder=clargs.output_folder)


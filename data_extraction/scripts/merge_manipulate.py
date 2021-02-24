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

# Use this script to merge data files in a folder (the opposite of split.py).
# The script will accept a folder containing all the JSON files, and it will
# merge them into a given file.

import sys
import argparse
import itertools

from data_extraction.data_reader.data_reader import Reader
from data_extraction.data_reader.manipulator.data_manipulator import DataManipulator
#from data_extraction.scripts.merge import Merger
import numpy as np
from multiprocessing import Pool

from utilities.basics import conditional_director_creator, dump_json


class MergerManipulator:
    def __init__(self, input_file_list=None,
                 output_folder=None,
                 dump_data_path=None,
                 num_chunks=32):
        # super().__init__(input_file_list, output_file)
        self.num_chunks = num_chunks
        self.output_folder = output_folder
        conditional_director_creator(self.output_folder)

        self.dump_data_path = dump_data_path
        self.chunked_file_list = [(_id, files) for _id, files in enumerate(self.split(input_file_list, chunks=self.num_chunks))]
        pool = Pool(processes=num_chunks)
        pool.map(self.read_a_chunk, self.chunked_file_list)

        self.full_reader = Reader(dump_data_path=dump_data_path)
        for chunk_id in range(num_chunks):
            file = self.output_folder + '/output_manipulated' + str(chunk_id) + '.json'
            self.full_reader.read_file(filename=file)
        self.full_reader.wrangle()
        self.full_reader.log_info()
        self.full_reader.dump()

    def split(self, input_file_list, chunks=None):
        with open(input_file_list, errors='ignore') as f:
            file_list = f.readlines()
        chunked_file_list = np.array_split(np.array(file_list), chunks)
        return chunked_file_list

    def read_a_chunk(self, chunk):
        chunk_id, chunk_files = chunk[0], chunk[1]
        output_file = self.output_folder + '/output_manipulated' + str(chunk_id) + '.json'
        self.read_and_dump(chunk_files, output_file, chunk_id=chunk_id)
        return

    def read_and_dump(self, file_list, output_file, chunk_id=None):
        mini_programs = self.read_one_chunked_file_list(file_list, chunk_id=chunk_id)
        dump_json({'size': len(mini_programs), 'programs': mini_programs}, output_file)

    def read_one_chunked_file_list(self, file_list, chunk_id=None):
        manipulator = DataManipulator(debug_print=False)
        mini_programs, j = [], 0
        for j, filename in enumerate(file_list):
            filename = filename[:-1]  # ignore '\n'
            try:
                manip_prog = manipulator.read_data(filename)
                mini_programs.extend(manip_prog)
                manipulator.reset()
            except:
                #print('Error merging file: {}'.format(filename))
                pass
            if (j+1) % 10000 == 0:
                print('Chunk id: {}, Loaded file: {}'.format(chunk_id, j+1), end='\n')

        print('Chunk id: {}, Loaded file: {}'.format(chunk_id, j+1), end='\n')
        return mini_programs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_list', type=str, nargs=1,
                        help='file containing list of all JSON files')
    parser.add_argument('--python_recursion_limit', type=int, default=100000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--data_path', type=str, default='data',
                        help='data to be saved here')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='file to output merged data')
    clargs = parser.parse_args()
    sys.setrecursionlimit(clargs.python_recursion_limit)
    MergerManipulator(input_file_list=clargs.file_list[0],
           output_folder=clargs.output_folder,
           dump_data_path=clargs.data_path)




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
import re
from itertools import chain

import numpy as np
import pickle

import wordninja
from nltk import WordNetLemmatizer

from data_extraction.data_reader.utils import split_camel, STOP_WORDS, gather_calls


class SurroundingReader:

    def __init__(self, max_elements=100, max_camel=3, vocab=None, infer=True):
        self.api_vocab, self.type_vocab, self.kw_vocab = vocab
        self.max_elements = max_elements
        self.max_camel = max_camel
        self.infer = infer

        self.surr_ret_types = None
        self.surr_fp_types = None
        self.surr_methods = None

        # TODO: remove this since already present in set reader
        self.lemmatizer = WordNetLemmatizer()

        return

    def read_surrounding(self, surrounding_methods):

        surr_ret_types = []
        surr_fp_types = []
        surr_method_names = []
        surr_method_ids = []

        for surr_method in surrounding_methods:
            return_type_id = self.type_vocab.conditional_add_or_get_node_val(surr_method['return_type'], self.infer)
            method_name = self.read_methodname(surr_method['surr_method'])
            formal_type_ids = []
            for formal in surr_method['formal_params']:
                fp_id = self.type_vocab.conditional_add_or_get_node_val(formal, self.infer)
                formal_type_ids.append(fp_id)
            surr_ret_types.append(return_type_id)
            surr_fp_types.append(formal_type_ids)
            surr_method_names.append(method_name)
            surr_method_ids.append(surr_method['id'])

        return surr_ret_types, surr_fp_types, surr_method_names, surr_method_ids

    # TODO: remove this since already present in set reader
    def read_methodname(self, name):
        name = name.split('@')[0]
        name_splits = self.split_words_underscore_plus_camel(name)
        kws = [self.kw_vocab.get_node_val(kw) if self.infer
                       else self.kw_vocab.conditional_add_node_val(kw)
                       for kw in name_splits]
        return kws

    # TODO: remove this since already present in set reader
    def split_words_underscore_plus_camel(self, s):

        # remove unicode
        s = s.encode('ascii', 'ignore').decode('unicode_escape', 'ignore')

        #remove numbers
        s = re.sub(r'\d+', '', s)
        #substitute all non alphabets by # to be splitted later
        s = re.sub("[^a-zA-Z]+", "#", s)
        #camel case split
        s = re.sub('(.)([A-Z][a-z]+)', r'\1#\2', s)  # UC followed by LC
        s = re.sub('([a-z0-9])([A-Z])', r'\1#\2', s)  # LC followed by UC
        vars = s.split('#')

        final_vars = []
        for var in vars:
            var = var.lower()
            words = wordninja.split(var)
            for word in words:
                w = self.lemmatizer.lemmatize(word, 'v')
                w = self.lemmatizer.lemmatize(w, 'n')
                len_w = len(w)
                if len_w > 1 and len_w < 10 and w not in STOP_WORDS:
                    final_vars.append(w)
        return final_vars

    # sz is total number of data points, Wrangle the set
    def wrangle(self, surr_ret_types, surr_fp_types, surr_methods, surr_method_ids, min_num_data=None):
        assert len(surr_ret_types) == len(surr_fp_types) == len(surr_methods)
        if min_num_data is None:
            sz = len(surr_ret_types)
        else:
            sz = max(min_num_data, len(surr_ret_types))

        self.surr_ret_types = np.zeros((sz, self.max_elements), dtype=np.int32)
        self.surr_fp_types = np.zeros((sz, self.max_elements, self.max_camel), dtype=np.int32)
        self.surr_methods = np.zeros((sz, self.max_elements, self.max_camel), dtype=np.int32)

        for i, all_ret_types in enumerate(surr_ret_types):
            for j, ret_type in enumerate(all_ret_types):
                if j >= self.max_elements:
                    continue
                self.surr_ret_types[i, surr_method_ids[i][j]] = ret_type

        for i, all_fp_types in enumerate(surr_fp_types):
            for j, fp_types in enumerate(all_fp_types):
                if j >= self.max_elements:
                    continue
                len_list = min(len(fp_types), self.max_camel)
                mod_list = fp_types[:len_list]
                self.surr_fp_types[i, surr_method_ids[i][j], :len_list] = mod_list

        for i, all_method_names in enumerate(surr_methods):
            for j, method_name in enumerate(all_method_names):
                if j >= self.max_elements:
                    continue
                len_list = min(len(method_name), self.max_camel)
                mod_list = method_name[:len_list]
                self.surr_methods[i, surr_method_ids[i][j], :len_list] = mod_list

        return

    def save(self, path):
        with open(path + '/surrounding.pickle', 'wb') as f:
            pickle.dump([self.surr_ret_types, self.surr_fp_types, self.surr_methods], f)
        return

    def load_data(self, path):
        with open(path + '/surrounding.pickle', 'rb') as f:
            self.surr_ret_types, self.surr_fp_types, self.surr_methods = pickle.load(f)
        return

    def truncate(self, sz):
        self.surr_ret_types = self.surr_ret_types[:sz, :self.max_elements]
        self.surr_fp_types = self.surr_fp_types[:sz, :self.max_elements, :self.max_camel]
        self.surr_methods = self.surr_methods[:sz, :self.max_elements, :self.max_camel]


    def split(self, num_batches):
        self.surr_ret_types = np.split(self.surr_ret_types, num_batches, axis=0)
        self.surr_fp_types = np.split(self.surr_fp_types, num_batches, axis=0)
        self.surr_methods = np.split(self.surr_methods, num_batches, axis=0)

    def get(self):
        return self.surr_ret_types, self.surr_fp_types, self.surr_methods

    def add_data_from_another_reader(self, surr_reader):
        if self.surr_ret_types is None:
            assert self.surr_fp_types is None and self.surr_methods is None
            self.surr_ret_types = surr_reader.surr_ret_types
            self.surr_fp_types = surr_reader.surr_fp_types
            self.surr_methods = surr_reader.surr_methods
        else:
            self.surr_ret_types = np.append(self.surr_ret_types, surr_reader.surr_ret_types, axis=0)
            self.surr_fp_types = np.append(self.surr_fp_types, surr_reader.surr_fp_types, axis=0)
            self.surr_methods = np.append(self.surr_methods, surr_reader.surr_methods, axis=0)

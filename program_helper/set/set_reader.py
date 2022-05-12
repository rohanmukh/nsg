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

from data_extraction.data_reader.utils import split_camel, STOP_WORDS


class SetReader:

    def __init__(self, max_elements=100, max_camel=3, vocab=None, infer=True):
        self.api_vocab, self.type_vocab, self.kw_vocab = vocab
        self.max_elements = max_elements
        self.max_camel = max_camel
        self.infer = infer

        self.apicalls = None
        self.types = None
        self.keywords = None
        self.method = None
        self.classname = None
        self.javadoc_kws = None

        self.lemmatizer = WordNetLemmatizer()
        # self.word2vecModel = gensim.models.KeyedVectors.load_word2vec_format('/rm38/GoogleNews-vectors-negative300.bin', binary=True)

        return

    def read_while_vocabing(self, program_ast):

        if 'apicalls' in program_ast:
            apicalls = [self.api_vocab.get_node_val(a) if self.infer else self.api_vocab.conditional_add_node_val(a)
                        for a in program_ast['apicalls']]
        else:
            apicalls = []

        if 'types' in program_ast:
            types = [self.type_vocab.get_node_val(t) if self.infer else self.type_vocab.conditional_add_node_val(t)
                     for t in program_ast['types']]
        else:
            types = []

        if 'keywords' in program_ast:
            keywords = [self.kw_vocab.get_node_val(kw) if self.infer else self.kw_vocab.conditional_add_node_val(kw)
                        for kw in program_ast['keywords']]
        else:
            keywords = []

        return apicalls, types, keywords

    def read_methodname(self, name):
        name = name.split('@')[0]
        return self.read_classname(name)

    def read_classname(self, name):
        name_splits = self.split_words_underscore_plus_camel(name)
        kws = [self.kw_vocab.get_node_val(kw) if self.infer
                       else self.kw_vocab.conditional_add_node_val(kw)
                       for kw in name_splits]
        return kws

    def read_natural_language(self, javadoc):
        if javadoc is None:
            return []
        javadoc = javadoc.strip()
        javadoc_list = self.split_words_underscore_plus_camel(javadoc)

        kws = [self.kw_vocab.get_node_val(kw) if self.infer
               else self.kw_vocab.conditional_add_node_val(kw)
               for kw in javadoc_list]
        return kws

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
    def wrangle(self, apicalls, types, keywords, method, classname, javadoc_kws, min_num_data=None):
        assert len(apicalls) == len(types) == len(keywords)
        if min_num_data is None:
            sz = len(apicalls)
        else:
            sz = max(min_num_data, len(apicalls))

        self.apicalls = np.zeros((sz, self.max_elements), dtype=np.int32)
        self.types = np.zeros((sz, self.max_elements), dtype=np.int32)
        self.keywords = np.zeros((sz, self.max_elements), dtype=np.int32)
        self.method = np.zeros((sz, self.max_camel), dtype=np.int32)
        self.classname = np.zeros((sz, self.max_camel), dtype=np.int32)
        self.javadoc_kws = np.zeros((sz, self.max_elements), dtype=np.int32)

        for i, a in enumerate(apicalls):
            len_list = min(len(a), self.max_elements)
            mod_list = a[:len_list]
            self.apicalls[i, :len_list] = mod_list

        for i, t in enumerate(types):
            len_list = min(len(t), self.max_elements)
            mod_list = t[:len_list]
            self.types[i, :len_list] = mod_list

        for i, kw in enumerate(keywords):
            len_list = min(len(kw), self.max_elements)
            mod_list = kw[:len_list]
            self.keywords[i, :len_list] = mod_list

        for i, m in enumerate(method):
            len_list = min(len(m), self.max_camel)
            mod_list = m[:len_list]
            self.method[i, :len_list] = mod_list

        for i, c in enumerate(classname):
            len_list = min(len(c), self.max_camel)
            mod_list = c[:len_list]
            self.classname[i, :len_list] = mod_list

        for i, j in enumerate(javadoc_kws):
            len_list = min(len(j), self.max_elements)
            mod_list = j[:len_list]
            self.javadoc_kws[i, :len_list] = mod_list

        return

    def save(self, path):
        with open(path + '/keywords.pickle', 'wb') as f:
            pickle.dump([self.apicalls, self.types, self.keywords,
                         self.method, self.classname, self.javadoc_kws], f)
        return

    def load_data(self, path):
        with open(path + '/keywords.pickle', 'rb') as f:
            self.apicalls, self.types, self.keywords,\
                self.method, self.classname, self.javadoc_kws = pickle.load(f)
        return

    def truncate(self, sz):
        self.apicalls = self.apicalls[:sz, :self.max_elements]
        self.types = self.types[:sz, :self.max_elements]
        self.keywords = self.keywords[:sz, :self.max_elements]
        self.method = self.method[:sz, :self.max_camel]
        self.classname = self.classname[:sz, :self.max_camel]
        self.javadoc_kws = self.javadoc_kws[:sz, :self.max_elements]

    def split(self, num_batches):
        self.apicalls = np.split(self.apicalls, num_batches, axis=0)
        self.types = np.split(self.types, num_batches, axis=0)
        self.keywords = np.split(self.keywords, num_batches, axis=0)
        self.method = np.split(self.method, num_batches, axis=0)
        self.classname = np.split(self.classname, num_batches, axis=0)
        self.javadoc_kws = np.split(self.javadoc_kws, num_batches, axis=0)

    def get(self):
        return self.apicalls, self.types, self.keywords, \
               self.method, self.classname, self.javadoc_kws


    def add_data_from_another_reader(self, keyword_reader):
        if self.apicalls is None:
            assert self.types is None and self.keywords is None
            assert self.method is None and self.javadoc_kws is None
            assert self.classname is None
            self.apicalls = keyword_reader.apicalls
            self.types = keyword_reader.types
            self.keywords = keyword_reader.keywords
            self.method = keyword_reader.method
            self.classname = keyword_reader.classname
            self.javadoc_kws = keyword_reader.javadoc_kws
        else:
            self.apicalls = np.append(self.apicalls, keyword_reader.apicalls, axis=0)
            self.types = np.append(self.types, keyword_reader.types, axis=0)
            self.keywords = np.append(self.keywords, keyword_reader.keywords, axis=0)
            self.method = np.append(self.method, keyword_reader.method, axis=0)
            self.classname = np.append(self.classname, keyword_reader.classname, axis=0)
            self.javadoc_kws = np.append(self.javadoc_kws, keyword_reader.javadoc_kws, axis=0)


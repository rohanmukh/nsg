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
import re
from itertools import chain

CONFIG_VOCAB = ['concept_dict', 'concept_dict_size',
                'api_dict', 'api_dict_size',
                'apiname_dict', 'apiname_dict_size',
                'type_dict', 'type_dict_size',
                'typename_dict', 'typename_dict_size',
                'var_dict', 'var_dict_size',
                'kw_dict', 'kw_dict_size',
                'op_dict', 'op_dict_size',
                'method_dict', 'method_dict_size'
                ]


# convert vocab to JSON
def dump_vocab(vocab):
    js = {}
    for attr in CONFIG_VOCAB:
        js[attr] = vocab.__getattribute__(attr)
    return js


def read_vocab(js):
    vocab = argparse.Namespace()
    for attr in CONFIG_VOCAB:
        vocab.__setattr__(attr, js[attr])

    chars_concept = dict()
    for item, value in vocab.concept_dict.items():
        chars_concept[value] = item
    vocab.__setattr__('chars_concept', chars_concept)

    chars_api = dict()
    for item, value in vocab.api_dict.items():
        chars_api[value] = item
    vocab.__setattr__('chars_api', chars_api)

    chars_apiname = dict()
    for item, value in vocab.apiname_dict.items():
        chars_apiname[value] = item
    vocab.__setattr__('chars_apiname', chars_apiname)

    chars_var = dict()
    for item, value in vocab.var_dict.items():
        chars_var[value] = item
    vocab.__setattr__('chars_var', chars_var)

    chars_type = dict()
    for item, value in vocab.type_dict.items():
        chars_type[value] = item
    vocab.__setattr__('chars_type', chars_type)

    chars_typename = dict()
    for item, value in vocab.typename_dict.items():
        chars_typename[value] = item
    vocab.__setattr__('chars_typename', chars_typename)

    chars_kw = dict()
    for item, value in vocab.kw_dict.items():
        chars_kw[value] = item
    vocab.__setattr__('chars_kw', chars_kw)

    chars_op = dict()
    for item, value in vocab.op_dict.items():
        chars_op[value] = item
    vocab.__setattr__('chars_op', chars_op)

    chars_method = dict()
    for item, value in vocab.method_dict.items():
        chars_method[value] = item
    vocab.__setattr__('chars_method', chars_method)

    return vocab


def gather_calls(node):
    """
    Gathers all call nodes (recursively) in a given AST node
    :param node: the node to gather calls from
    :return: list of call nodes
    """

    if type(node) is list:
        return list(chain.from_iterable([gather_calls(n) for n in node]))
    node_type = node['node']
    if node_type == 'DSubTree':
        return gather_calls(node['_nodes'])
    elif node_type == 'DBranch':
        return gather_calls(node['_cond']) + gather_calls(node['_then']) + gather_calls(node['_else'])
    elif node_type == 'DExcept':
        return gather_calls(node['_try']) + gather_calls(node['_catch'])
    elif node_type == 'DLoop':
        return gather_calls(node['_cond']) + gather_calls(node['_body'])
    elif node_type == 'DInfix':
        return gather_calls(node['_left']) + gather_calls(node['_right'])
    elif node_type == 'DAPIInvoke':  # this node itself is a call
        return node['_calls']
    else:
        return []

def gather_extra_infos(node):
    """
    Gathers all call nodes (recursively) in a given AST node
    :param node: the node to gather calls from
    :return: list of call nodes
    """

    if type(node) is list:
        return list(chain.from_iterable([gather_extra_infos(n) for n in node]))
    node_type = node['node']
    if node_type == 'DSubTree':
        return gather_extra_infos(node['_nodes'])
    elif node_type == 'DBranch':
        return ['conditional'] + gather_extra_infos(node['_cond']) + gather_extra_infos(node['_then']) + gather_extra_infos(node['_else'])
    elif node_type == 'DExcept':
        return ['exception'] + gather_extra_infos(node['_try']) + gather_extra_infos(node['_catch'])
    elif node_type == 'DLoop':
        return ['loop'] + gather_extra_infos(node['_cond']) + gather_extra_infos(node['_body'])
    elif node_type == 'DInfix':
        return ['infix'] + gather_extra_infos(node['_left']) + gather_extra_infos(node['_right'])
    else:
        return []


def split_camel(s):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1#\2', s)  # UC followed by LC
    s = re.sub('([a-z0-9])([A-Z])', r'\1#\2', s)  # LC followed by UC
    return s.split('#')


STOP_WORDS = {  # CoreNLP English stop words
    "'ll", "'s", "'m", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me",
    "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "return", "arent", "cant", "couldnt", "didnt", "doesnt",
    "dont", "hadnt", "hasnt", "havent", "hes", "heres", "hows", "im", "isnt", "its", "lets", "mustnt",
    "shant", "shes", "shouldnt", "thats", "theres", "theyll", "theyre", "theyve", "wasnt", "were",
    "werent", "whats", "whens", "wheres", "whos", "whys", "wont", "wouldnt", "youd", "youll", "youre",
    "youve"
}
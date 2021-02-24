import argparse
import sys

import os

import json

from data_extraction.data_reader.utils import read_vocab

def check_angle_match(type):
    count = 0
    for s in type:
        if s == '<':
            count += 1
        elif s == '>':
            count -= 1
    return count == 0

def test_config_types(clargs):
    config_file = os.path.join(clargs.data, 'vocab.json')
    with open(config_file) as f:
        config = read_vocab(json.load(f))

    for j, type in enumerate(config.type_dict.keys()):
        if not check_angle_match(type):
            print(type)
            assert False
    print('\nSuccess with {} types'.format(j))

    for j, api in enumerate(config.api_dict.keys()):
        if not check_angle_match(api):
            print(api)
            assert False
    print('\nSuccess with {} apis'.format(j))


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--python_recursion_limit', type=int, default=10000,
                        help='set recursion limit for the Python interpreter')
    parser.add_argument('--data', type=str, default='../data_extraction/data_reader/data',
                        help='load data from here')
    clargs_ = parser.parse_args()
    sys.setrecursionlimit(clargs_.python_recursion_limit)

    test_config_types(clargs_)

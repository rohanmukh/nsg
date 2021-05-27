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

import json
import os
import re

def stripJavaDoc(stringBody):
    return re.sub(r'/\*\*(.*?)\*\/', '', stringBody.replace('\n',''))

def dictOfset2list(inp_dict_of_set):
    for key, value in inp_dict_of_set.items():
        inp_dict_of_set[key] = list(value)
    return inp_dict_of_set


def dump_json(js, path):
    with open(path, 'w') as f:
        json.dump(js, fp=f, indent=2)


def read_json(data_path):
    with open(data_path) as f:
        js_data = json.load(f)
    return js_data



def dump_file(data, path):
    with open(path, 'w') as f:
        f.write(data)


def dump_java(java_progs, path, real_codes=None):
    f = open(path, 'w')
    for i, java_beams in enumerate(java_progs):
        if real_codes is not None:
            f.write("\n-------REAL CODE--------\n")
            real_code = real_codes[i]
            f.write(real_code.encode('utf-16','surrogatepass').decode('utf-16'))
            f.write("\n---------------\n")

        f.write("\n------START OF PREDICTION---------\n")
        for java in java_beams:
            f.write(java.encode('utf-16','surrogatepass').decode('utf-16'))
            f.write("\n---------------\n")
        f.write("\n------END OF BEAM SEARCH---------\n\n\n")

def conditional_director_creator(path):
    if not os.path.exists(path):
        os.makedirs(path)


def truncate_two_decimals(value):
    return float(int(value*100))/100.

def reconstruct_camel_case(arr_kws):
    return ''.join([kw.title() if i>0 else kw for i, kw in enumerate(arr_kws)])
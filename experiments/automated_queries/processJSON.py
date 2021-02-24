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
import sys
import random
import os
from copy import deepcopy

from data_extraction.data_reader.manipulator.data_manipulator import DataManipulator
from utilities.basics import dump_json, conditional_director_creator
from utilities.vocab_building_dictionary import DELIM

max_ast_depth = 32
num_of_exps = 8


def processJSONs(inFile, logdir):
    random.seed(12)
    processor = DataManipulator(debug_print=False)
    print("Processing JSONs ... ", end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
          os.makedirs(logdir)

    with open(inFile) as f:
        jsonLines = f.readlines()

    programs = [[] for i in range(num_of_exps)]
    count = 0
    for j, line in enumerate(jsonLines):
        line = line.strip()
        if os.path.isfile(line):
            list_of_js = processEachJSON(processor, line)
            for i in range(num_of_exps):
                if list_of_js[i] is not None:
                    programs[i].append(list_of_js[i])
        print('Number of lines processed {}'.format(j), end='\r')

    for expNumber in range(num_of_exps):
        exp_logdir = logdir + "/expNumber_" + str(expNumber)
        conditional_director_creator(exp_logdir)
        dump_json({'programs': programs[expNumber]}, os.path.join(exp_logdir, 'L4TestProgramList.json'))

    print("Done")
    return count


def processEachJSON(processor, fileName):
    js_list = processor.read_data(fileName)
    processor.reset()
    random.shuffle(js_list)
    if len(js_list) == 0:
        return [None for _ in range(num_of_exps)]
    js = js_list[0]
    #
    # js = json.JSONDecoder().decode(js)

    list_of_js = [None for i in range(num_of_exps)]
    for i in range(num_of_exps):
        js_t = deepcopy(js)
        js_t = modifyInputForExperiment(js_t, i)
        if js_t != {}:
            list_of_js[i] = js_t

    return list_of_js


def modifyInputForExperiment(sample, expNumber):


    if ( 'apicalls' not in sample ) or ('apicalls' in sample and len(sample['apicalls']) < 1):
         return {}

    ## You need to have all sorrounding infos
    for ev in ['javaDoc', 'Surrounding_Methods',
               'return_type', 'formal_params', 'apicalls', 'types', 'keywords']:
        if ev not in sample:
            return {}
        if ev == 'javaDoc' and (sample[ev] is None or len(sample[ev].split(" ")) < 3):
            return {}
        if ev == 'Surrounding_Methods' and len(sample[ev]) < 1:
            return {}


    if expNumber == 0: ##  only method header
        for ev in ['apicalls', 'types', 'keywords', 'className', 'javaDoc', 'Surrounding_Methods']:
            if ev in sample:
                del sample[ev]
        sample['className'] = DELIM
        sample['javaDoc'] = None
        sample['Surrounding_Methods'] = []

    elif expNumber == 1: ## method header + surrounding infor
        for ev in ['apicalls', 'types', 'keywords', 'javaDoc']:
            if ev in sample:
                del sample[ev]
        sample['javaDoc'] = None

    elif expNumber == 2: ## method header + surrounding info + javadoc
        for ev in ['apicalls', 'types', 'keywords']:
            if ev in sample:
                del sample[ev]

    elif expNumber == 3: ## method header + surrounding info + javadoc + kw
        for ev in ['apicalls', 'types']:
            if ev in sample:
                del sample[ev]

    elif expNumber == 4: ##  all evidence, visib = 100%
        pass

    elif expNumber == 5: ##  till method header
        visibility = 0.75
        evidence_subsample(sample, visibility)

    elif expNumber == 6: ##  till method header
        visibility = 0.5
        evidence_subsample(sample, visibility)

    elif expNumber == 7: ##  till method header
        visibility = 0.25
        evidence_subsample(sample, visibility)


    return sample


def evidence_subsample(sample, visibility):
    def subsample_list(list_data):
        output = []
        for val in list_data:
            if random.random() < visibility:
                output.append(val)
        return output

    def subsample_string(inp_str):
        str_list = inp_str.strip().split(" ")
        return " ".join(subsample_list(str_list))

    def subsample_element(element):
        if random.random() < visibility:
            return element
        return DELIM

    for ev in ['apicalls', 'types', 'keywords', 'Surrounding_Methods']:
        sample[ev] = subsample_list(sample[ev])

    sample['javaDoc'] = subsample_string(sample['javaDoc'])
    for ev in ['method', 'className']:
        sample[ev] = subsample_element(sample[ev])


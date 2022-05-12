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

DELIM = "__delim__"

class VocabBuildingDictionary:
    def __init__(self, preset_dict=None):
        if preset_dict is None:
            self.call_dict = dict()
            self.call_dict[DELIM] = self.dict_value_format(0)
            self.call_count = 1
        else:
            self.call_dict = preset_dict
            self.call_count = len(preset_dict)

    def load_dictionary(self, pre_loaded_vocab):
        self.call_dict = pre_loaded_vocab.vocab
        self.call_count = pre_loaded_vocab.vocab_size


    def conditional_add_or_get_node_val(self, node_val, infer):
        if infer is True:
            return self.get_node_val(node_val)
        else:
            return self.conditional_add_node_val(node_val)


    def conditional_add_node_val(self, node_val):
        if node_val in self.call_dict:
            return self.call_dict[node_val]
        else:
            next_open_pos = self.call_count
            self.call_dict[node_val] = self.dict_value_format(next_open_pos, node_val)
            self.call_count += 1
            return next_open_pos

    def get_node_val(self, node_val):
        if node_val not in self.call_dict:
            return 0
        else:
            return self.call_dict[node_val]

    def get_dictionary(self):
        return self.call_dict, self.call_count

    def dict_value_format(self, id, key=None):
        return id


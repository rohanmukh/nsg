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
from data_extraction.data_reader.utils import read_vocab, dump_vocab

CONFIG_GENERAL = ['batch_size', 'num_epochs', 'latent_size',
                  'ev_drop_rate', "ev_miss_rate", "decoder_drop_rate",
                  'learning_rate', 'max_ast_depth', 'input_fp_depth',
                  'max_keywords', 'max_variables', 'max_fields', 'max_camel_case',
                  'trunct_num_batch', 'print_step', 'checkpoint_step', ]
CONFIG_ENCODER = ['units', 'num_layers']
CONFIG_DECODER = ['units', 'num_layers', 'ifnag']



# convert JSON to config
def read_config(js, infer=False):
    config = argparse.Namespace()

    for attr in CONFIG_GENERAL:
        config.__setattr__(attr, js[attr])

    config.encoder = argparse.Namespace()
    for attr in CONFIG_ENCODER:
        config.encoder.__setattr__(attr, js['encoder'][attr])

    config.decoder = argparse.Namespace()
    for attr in CONFIG_DECODER:
        config.decoder.__setattr__(attr, js['decoder'][attr])

    if infer:
        config.vocab = read_vocab(js['vocab'])

    return config


# convert config to JSON
def dump_config(config):
    js = {}

    for attr in CONFIG_GENERAL:
        js[attr] = config.__getattribute__(attr)

    js['encoder'] = {attr: config.encoder.__getattribute__(attr) for attr in
                     CONFIG_ENCODER}

    js['decoder'] = {attr: config.decoder.__getattribute__(attr) for attr in
                     CONFIG_DECODER}

    js['vocab'] = dump_vocab(config.vocab)

    return js


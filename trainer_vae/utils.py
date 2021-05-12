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

EXPANSION_LABELED_EDGE_TYPE_NAMES = []
EXPANSION_UNLABELED_EDGE_TYPE_NAMES = ["Child", "Parent", "NextSibling",
                                       "InheritedToSynthesised",
                                       "NextToken", "NextUse"]



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
        try:
            config.decoder.__setattr__(attr, js['decoder'][attr])
        except:
            continue

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


def prepare_gnn2nag_data(batch_data, model):
    gnn_feed_dict = {}
    total_edge_types = len(EXPANSION_LABELED_EDGE_TYPE_NAMES) + len(
        EXPANSION_UNLABELED_EDGE_TYPE_NAMES)
    #flat_batch_keys = ['eg_node_token_ids',
    #                   'eg_initial_node_ids',
    #                   'eg_receiving_node_nums']
    flat_batch_keys = ['eg_node_token_ids',
                       'eg_initial_node_ids',
                       'eg_receiving_node_nums']
    for key in flat_batch_keys:
        write_to_minibatch(gnn_feed_dict,
                           model.placeholders[key],
                           batch_data[key])
    for step in range(model.hyperparameters['eg_propagation_substeps']):
        write_to_minibatch(
            gnn_feed_dict,
            model.placeholders['eg_msg_target_node_ids'][step],
            np.concatenate(batch_data['eg_msg_target_node_ids'][step]))
        write_to_minibatch(
            gnn_feed_dict,
            model.placeholders['eg_receiving_node_ids'][step],
            batch_data['eg_receiving_node_ids'][step])
        for edge_idx in range(total_edge_types):
            write_to_minibatch(
                gnn_feed_dict,
                model.placeholders['eg_sending_node_ids'][step][edge_idx],
                batch_data['eg_sending_node_ids'][step][edge_idx])
    return gnn_feed_dict


def write_to_minibatch(minibatch, placeholder, val):
    if len(val) == 0:
        ph_shape = placeholder.shape.as_list()
        ph_shape[0] = 0
        minibatch[placeholder] = np.empty(ph_shape)
    else:
        minibatch[placeholder] = np.array(val)


def init_mini_batch(batch_data, model):
    total_edge_types = len(EXPANSION_UNLABELED_EDGE_TYPE_NAMES) + len(
        EXPANSION_LABELED_EDGE_TYPE_NAMES)
    eg_propagation_substeps = model.hyperparameters['eg_propagation_substeps']
    batch_data['eg_node_offset'] = 0
    batch_data['eg_node_token_ids'] = []
    batch_data['eg_initial_node_ids'] = []
    batch_data['eg_sending_node_ids'] = [[[] for _ in range(
        total_edge_types)] for _ in range(eg_propagation_substeps)]
    batch_data['next_step_target_node_id'] = \
        [0 for _ in range(eg_propagation_substeps)]
    batch_data['eg_msg_target_node_ids'] = \
        [[[] for _ in range(total_edge_types)] for _ in range(
            eg_propagation_substeps)]
    batch_data['eg_receiving_node_ids'] = [[] for _ in range(
        eg_propagation_substeps)]
    batch_data['eg_receiving_node_nums'] = [0 for _ in range(
        eg_propagation_substeps)]


def extend_batch_data(batch_data, gnn_info, model):
    eg_schedule = gnn_info['eg_schedule']
    #node_labels = gnn_info['node_labels']
    node_ids = gnn_info['node_ids']
    batch_data['eg_node_token_ids'].extend(node_ids)
    total_edge_types = len(EXPANSION_UNLABELED_EDGE_TYPE_NAMES) + len(
        EXPANSION_LABELED_EDGE_TYPE_NAMES)
    eg_propagation_substeps = model.hyperparameters['eg_propagation_substeps']

    num_eg_nodes = len(node_ids)
    for eg_node_id in range(num_eg_nodes):
        batch_data['eg_initial_node_ids'].append(
            eg_node_id + batch_data['eg_node_offset'])

    len_path = min(len(eg_schedule), eg_propagation_substeps - 1)
    try:
        for (step_num, schedule_step) in enumerate(eg_schedule[:len_path]):
            eg_node_id_to_step_target_id = OrderedDict()
            for edge_type in range(total_edge_types):
                for (source, target) in schedule_step[edge_type]:
                    batch_data['eg_sending_node_ids'][step_num][edge_type].append(
                        source + batch_data['eg_node_offset'])
                    step_target_id = eg_node_id_to_step_target_id.get(target)
                    if step_target_id is None:
                        step_target_id = batch_data['next_step_target_node_id'][step_num]
                        batch_data['next_step_target_node_id'][step_num] += 1
                        eg_node_id_to_step_target_id[target] = step_target_id
                    batch_data['eg_msg_target_node_ids'][step_num][edge_type].append(
                        step_target_id)
            for eg_target_node_id in eg_node_id_to_step_target_id.keys():
                batch_data['eg_receiving_node_ids'][step_num].append(
                    eg_target_node_id + batch_data['eg_node_offset'])
            batch_data['eg_receiving_node_nums'][step_num] += len(
                eg_node_id_to_step_target_id)
    except:
        import pdb; pdb.set_trace()

    batch_data['eg_node_offset'] += len(node_ids)


def construct_minibatch(batch_gnn_info, model):
    batch_data = {}
    init_mini_batch(batch_data, model)
    for gnn_info in batch_gnn_info:
        extend_batch_data(batch_data, gnn_info, model)
    gnn_minibatch = prepare_gnn2nag_data(batch_data, model)
    return gnn_minibatch, batch_data

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

import tensorflow as tf
from program_helper.sequence.simple_lstm import SimpleLSTM


class SequenceDecoder(SimpleLSTM):
    def __init__(self, num_layers, units,
                 fp_nodes, fp_type_or_not,
                 initial_state, ret_type, batch_size,
                 type_vocab_size,
                 concept_vocab_size,
                 type_emb, concept_emb):

        super().__init__(units, num_layers, output_units=units)
        self.type_emb = type_emb
        self.concept_emb = concept_emb
        self.initialize_lstms(units, num_layers, concept_vocab_size, type_vocab_size)

        with tf.variable_scope('seq_decoder'):
            self.state = initial_state
            type_output_logits, \
                concept_output_logits = [], []
            for i in range(len(fp_nodes)):
                if i > 0:
                    tf.compat.v1.get_variable_scope().reuse_variables()

                self.state, output, logits = self.get_next_output_for_fp(fp_nodes[i],
                                                                         fp_type_or_not[i],
                                                                         ret_type,
                                                                         self.state)

                concept_logit, type_logit = logits

                type_output_logits.append(type_logit)
                concept_output_logits.append(concept_logit)

        self.output_logits = [
            tf.stack(concept_output_logits, 1),
            tf.ones((batch_size, len(fp_nodes), batch_size), dtype=tf.float32),
            tf.stack(type_output_logits, 1),
            tf.ones((batch_size, len(fp_nodes), batch_size), dtype=tf.float32),
            tf.ones((batch_size, len(fp_nodes), batch_size), dtype=tf.float32),
            tf.ones((batch_size, len(fp_nodes), batch_size), dtype=tf.float32)
        ]

    def initialize_lstms(self, units, num_layers, concept_vocab_size, type_vocab_size):
        with tf.variable_scope('concept_prediction'):
            self.concept_encoder = SimpleLSTM(units, num_layers,
                                              output_units=concept_vocab_size)

        with tf.variable_scope('type_prediction'):
            self.type_encoder = SimpleLSTM(units, num_layers,
                                           output_units=type_vocab_size)

    def get_next_output_for_fp(self, node,
                               type_or_not, ret_type,
                               state_in):
        with tf.variable_scope('concept_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               tf.nn.embedding_lookup(self.type_emb, ret_type)],
                              axis=1)
            concept_output, concept_state = self.concept_encoder.get_next_output_with_symtab(input,
                                                                                             state_in)
            concept_logit = self.concept_encoder.get_projection(concept_output)

        with tf.variable_scope('type_prediction'):
            input = tf.concat([tf.nn.embedding_lookup(self.concept_emb, node),
                               tf.nn.embedding_lookup(self.type_emb, ret_type)],
                              axis=1)
            type_output, type_state = self.type_encoder.get_next_output_with_symtab(input,
                                                                                    state_in)
            type_logit = self.type_encoder.get_projection(type_output)

        output = tf.where(type_or_not, type_output, concept_output)
        state = [tf.where(type_or_not, type_state[j], concept_state[j]) for j in range(self.num_layers)]
        logits = [concept_logit, type_logit ]
        return state, output, logits

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

from program_helper.element.dense_encoder import DenseEncoder
from program_helper.sequence.sequence_encoder import SequenceEncoder
from program_helper.set.set_encoder import KeywordEncoder
from program_helper.surrounding.surrounding_encoder import SurroundingEncoder


class ProgramEncoder:

    def __init__(self, config,
                 apicalls, types, keywords,
                 fp_inputs, field_inputs,
                 ret_type,
                 method, classname, javadoc,
                 surr_ret, surr_fp, surr_method,
                 ev_drop_rate=None,
                 ev_miss_rate=None
                 ):

        self.type_emb = tf.get_variable('emb_type', [config.vocab.type_dict_size, config.encoder.units])

        self.apiname_emb = tf.get_variable('emb_apiname', [config.vocab.api_dict_size, config.encoder.units])
        self.typename_emb = tf.get_variable('emb_typename', [config.vocab.type_dict_size, config.encoder.units])
        self.kw_emb = tf.get_variable('emb_kw', [config.vocab.kw_dict_size, config.encoder.units])

        '''
            let us unstack the drop prob 
        '''
        ev_drop_rate = tf.unstack(ev_drop_rate)
        with tf.variable_scope("ast_tree_api"):
            self.ast_mean_api = KeywordEncoder(apicalls,
                                               config.encoder.units, config.encoder.num_layers,
                                               config.vocab.apiname_dict_size,
                                               config.batch_size,
                                               config.max_keywords,
                                               emb=self.apiname_emb,
                                               drop_prob=ev_drop_rate[0]
                                               )
            ast_mean_api = self.ast_mean_api.output

        with tf.variable_scope("ast_tree_types"):
            self.ast_mean_types = KeywordEncoder(types,
                                                 config.encoder.units, config.encoder.num_layers,
                                                 config.vocab.typename_dict_size,
                                                 config.batch_size,
                                                 config.max_keywords,
                                                 emb=self.typename_emb,
                                                 drop_prob=ev_drop_rate[1]
                                                 )
            ast_mean_types = self.ast_mean_types.output
        with tf.variable_scope("ast_tree_kw"):
            self.ast_mean_kws = KeywordEncoder(keywords,
                                               config.encoder.units, config.encoder.num_layers,
                                               config.vocab.kw_dict_size,
                                               config.batch_size,
                                               config.max_keywords,
                                               emb=self.kw_emb,
                                               drop_prob=ev_drop_rate[2]
                                               )
            ast_mean_kw = self.ast_mean_kws.output

        with tf.variable_scope("formal_param"):
            self.fp_mean_enc = KeywordEncoder( fp_inputs,
                                               config.encoder.units, config.encoder.num_layers,
                                               config.vocab.type_dict_size,
                                               config.batch_size,
                                               config.input_fp_depth,
                                               emb=self.type_emb,
                                               drop_prob=ev_drop_rate[3]
                                               )
            fp_mean = self.fp_mean_enc.output

        with tf.variable_scope("field_vars"):
            self.field_enc = KeywordEncoder(field_inputs,
                                             config.encoder.units, config.encoder.num_layers,
                                             config.vocab.type_dict_size,
                                             config.batch_size,
                                             config.max_fields,
                                             emb=self.type_emb,
                                             drop_prob=ev_drop_rate[4]
                                            )
            field_mean = self.field_enc.output

        with tf.variable_scope("ret_type"):
            self.ret_mean_enc = DenseEncoder(ret_type,
                                             config.encoder.units, config.encoder.num_layers,
                                             config.latent_size,
                                             config.vocab.type_dict_size, config.batch_size,
                                             emb=self.type_emb,
                                             drop_prob=ev_drop_rate[5]
                                             )
            ret_mean = self.ret_mean_enc.latent_encoding

        with tf.variable_scope("method_kw"):
            self.method_enc = KeywordEncoder(method,
                                             config.encoder.units, config.encoder.num_layers,
                                             config.vocab.kw_dict_size,
                                             config.batch_size,
                                             config.max_camel_case,
                                             emb=self.kw_emb,
                                             drop_prob=ev_drop_rate[6]
                                             )
            method_mean_kw = self.method_enc.output

        with tf.variable_scope("class_kw"):
            self.class_enc = KeywordEncoder(classname,
                                            config.encoder.units, config.encoder.num_layers,
                                            config.vocab.kw_dict_size,
                                            config.batch_size,
                                            config.max_camel_case,
                                            emb=self.kw_emb,
                                            drop_prob=ev_drop_rate[7]
                                            )
            class_mean_kw = self.class_enc.output

        with tf.variable_scope("javadoc_kw"):
            self.jd_enc = KeywordEncoder(javadoc,
                                         config.encoder.units, config.encoder.num_layers,
                                         config.vocab.kw_dict_size,
                                         config.batch_size,
                                         config.max_keywords,
                                         emb=self.kw_emb,
                                         drop_prob=ev_drop_rate[8]
                                         )
            jd_mean = self.jd_enc.output

        with tf.variable_scope("surrounding"):
            self.surr_enc = SurroundingEncoder(surr_ret, surr_fp, surr_method,
                                         config.encoder.units, config.encoder.num_layers,
                                         config.vocab.type_dict_size,
                                         config.vocab.kw_dict_size,
                                         config.batch_size,
                                         config.max_keywords,
                                         config.max_camel_case,
                                         self.type_emb,
                                         self.kw_emb,
                                         drop_prob=ev_drop_rate[9]
                                         )
            surr_mean = self.surr_enc.output

        evidences = [ast_mean_api, ast_mean_types, ast_mean_kw,
                     fp_mean, ret_mean,
                     field_mean,
                     method_mean_kw, class_mean_kw, jd_mean, surr_mean,
                     ]

        '''
        Lets drop some evidence types altogether according to #ev_miss_rate
        '''
        evidences = [tf.where(tf.random_uniform((config.batch_size, config.latent_size), 0, 1, dtype=tf.float32) > ev_miss_rate[j],
                              ev,
                              tf.zeros_like(ev)) for j, ev in enumerate(evidences)]


        with tf.variable_scope('sigma'):
            sigmas = tf.get_variable('sigma', [len(evidences)])
            sigmas = tf.unstack(sigmas)

        d = [tf.where(tf.not_equal(tf.reduce_sum(ev, axis=1), 0.),
                      tf.tile([1. / tf.square(sigma)], [config.batch_size]),
                      tf.zeros(config.batch_size)) for ev, sigma in zip(evidences, sigmas)]
        d = 1. + tf.reduce_sum(tf.stack(d), axis=0)
        denom = tf.tile(tf.reshape(d, [-1, 1]), [1, config.latent_size])

        encodings = [ev / tf.square(sigma) for ev, sigma in
                     zip(evidences, sigmas)]
        encodings = [tf.where(tf.not_equal(tf.reduce_sum(enc, axis=1), 0.),
                                   enc,
                                   tf.zeros([config.batch_size, config.latent_size], dtype=tf.float32)
                                   ) for enc in encodings]


        self.sigmas = sigmas
        self.mean = tf.reduce_sum(tf.stack(encodings, axis=0), axis=0) / denom
        I = tf.ones([config.batch_size, config.latent_size], dtype=tf.float32)
        self.covar = I / denom


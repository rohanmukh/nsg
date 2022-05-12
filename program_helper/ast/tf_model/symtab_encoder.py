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


class SymTabEncoder:
    def __init__(self, units, num_layers,
                 num_vars=None,
                 batch_size=None,
                 drop_prob=None):

        # +1 for empty 0-th entry which is responsible for putsting default vars like system type
        self.num_vars = num_vars #+ 1
        self.units = units

        self.batch_size = batch_size

        return

    def create_symtab(self, batch_size, units):
        return tf.zeros((batch_size, self.num_vars, units), dtype=tf.float32)


    def update_symtab(self, type_input, var_id, inp_symtab):
        valid_vars = tf.reshape(tf.one_hot(var_id, self.num_vars), (-1,))
        type_embedding = tf.reshape(tf.tile(type_input, (1, self.num_vars)), (-1, self.units))
        type_embedding = tf.where(tf.not_equal(valid_vars, 0), type_embedding, tf.zeros_like(type_embedding))

        symtab = tf.reshape(inp_symtab, (-1, self.units))
        symtab = symtab + type_embedding

        symtab = tf.where(tf.not_equal(type_embedding, 0), symtab, tf.reshape(inp_symtab, (-1, self.units)))

        # symtab is BS * NV * dims
        symtab = tf.reshape(symtab, (-1, self.num_vars, self.units))

        return symtab

    def strip_symtab(self, var_decl_id, symtab):
        var_range = tf.tile(tf.expand_dims(tf.range(0, self.num_vars), 0), [self.batch_size, 1])
        symtab_mask = tf.tile(tf.expand_dims(var_decl_id, 1), [1, self.num_vars]) >= var_range
        symtab_mask = tf.tile(tf.expand_dims(symtab_mask, 2), [1, 1, self.units])
        stripped_symtab = tf.where(symtab_mask, symtab, tf.zeros_like(symtab))
        return stripped_symtab


    def create_unused_varflag(self, batch_size):
        return tf.zeros((batch_size, self.num_vars), dtype=tf.float32)

    def decl_update_unused_vars(self, var_id, inp_unused_varflag):
        valid_vars = tf.one_hot(var_id, self.num_vars)
        inp_unused_varflag = tf.where(tf.equal(valid_vars, 1), tf.ones_like(inp_unused_varflag), inp_unused_varflag )
        return inp_unused_varflag

    def usage_update_unused_vars(self, var_id, inp_unused_varflag):
        valid_vars = tf.one_hot(var_id, self.num_vars)
        inp_unused_varflag = tf.where(tf.equal(valid_vars, 1), tf.zeros_like(inp_unused_varflag), inp_unused_varflag )
        return inp_unused_varflag


    def strip_unused_vars(self, var_decl_id, symtab):
        var_range = tf.tile(tf.expand_dims(tf.range(0, self.num_vars), 0), [self.batch_size, 1])
        symtab_mask = tf.tile(tf.expand_dims(var_decl_id, 1), [1, self.num_vars]) >= var_range
        stripped_symtab = tf.where(symtab_mask, symtab, tf.zeros_like(symtab))
        return stripped_symtab

    def create_nullptr_varflag(self, batch_size):
        return tf.zeros((batch_size, self.num_vars), dtype=tf.float32)

    def decl_update_nullptr_varflag(self, var_id, inp_unused_varflag):
        return self.decl_update_unused_vars(var_id, inp_unused_varflag)

    def usage_update_nullptr_varflag(self, var_id, inp_unused_varflag):
        return self.usage_update_unused_vars(var_id, inp_unused_varflag)

    def strip_nullptr_varflag(self, var_decl_id, symtab):
        return self.strip_unused_vars(var_decl_id, symtab)

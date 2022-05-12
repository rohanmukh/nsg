import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_" \
                                  "BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def permute_tensor(num_vars, batch_size, var_decl_id, units, symtab):
    var_range = tf.tile(tf.expand_dims(tf.range(0, num_vars), 0), [batch_size, 1])
    symtab_mask = tf.tile(tf.expand_dims(var_decl_id, 1), [1, num_vars]) >= var_range
    symtab_mask = tf.tile(tf.expand_dims(symtab_mask, 2), [1, 1, units])
    stripped_symtab = tf.where(symtab_mask, symtab, tf.zeros_like(symtab))
    return stripped_symtab

batch_size = 4
num_vars = 10
units = 32

symtab = tf.random.normal((batch_size, num_vars, units))
np_var_decl_id = np.random.randint(num_vars, size=batch_size)
var_decl_id = tf.Variable(np_var_decl_id, dtype=tf.int32)

stripped_symtab = permute_tensor(num_vars, batch_size, var_decl_id, units, symtab)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
[y, y_strip] = sess.run([symtab, stripped_symtab])

print(y)
print(y_strip)


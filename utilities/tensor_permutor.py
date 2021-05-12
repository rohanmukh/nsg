import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_" \
                                  "BUS_ID"  # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf


def permute_batched_tensor_3dim(batched_x, batched_perm_ids):
    indices = tf.tile(tf.expand_dims(batched_perm_ids, 2), [1,1,batched_x.shape[2]])

    # Create additional indices
    i1, i2 = tf.meshgrid(tf.range(batched_x.shape[0]),
                         tf.range(batched_x.shape[2]), indexing="ij")
    i1 = tf.tile(i1[:, tf.newaxis, :], [1, batched_x.shape[1], 1])
    i2 = tf.tile(i2[:, tf.newaxis, :], [1, batched_x.shape[1], 1])
    # Create final indices
    idx = tf.stack([i1, indices, i2], axis=-1)
    temp = tf.scatter_nd(idx, batched_x, batched_x.shape)
    return temp

def permute_batched_tensor_2dim(batched_x, batched_perm_ids):
    batched_x = tf.expand_dims(batched_x, axis=-1)
    out = permute_batched_tensor_3dim(batched_x, batched_perm_ids)
    return tf.squeeze(out)



def test_permute_tensor_batched_3d():
    batch_size = 8
    num_vars = 10
    dims = 3
    x = tf.random.normal((batch_size, num_vars, dims))
    np_perm_id = np.vstack([np.random.permutation(num_vars) for i in range(batch_size)])
    tf_perm_id = tf.Variable(np_perm_id, dtype=tf.int32)

    x_perm = permute_batched_tensor_3dim(x, tf_perm_id)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    [y, y_perm] = sess.run([x, x_perm])

    for i in range(batch_size):
        print("X was")
        print(y[i])
        print("Permuted it becomes")
        print(y_perm[i])
        print("Ids were")
        print(np_perm_id[i])

    for i in range(batch_size):
        for y_i, y_perm_i, id in zip(y, y_perm, np_perm_id):
            for i, id_i in enumerate(id):
                for m, n in zip(y_perm_i[id_i], y_i[i]):
                    assert m == n





def test_permute_tensor_batched_2d():
    batch_size = 8
    num_vars = 10
    x = tf.random.normal((batch_size, num_vars))
    np_perm_id = np.vstack([np.random.permutation(num_vars) for i in range(batch_size)])
    tf_perm_id = tf.Variable(np_perm_id, dtype=tf.int32)

    x_perm = permute_batched_tensor_2dim(x, tf_perm_id)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    [y, y_perm] = sess.run([x, x_perm])

    print("X was")
    print(y)
    print("Permuted it becomes")
    print(y_perm)
    print("Ids were")
    print(np_perm_id)

    # for y_i, y_perm_i, id in zip(y, y_perm, np_perm_id):
    #     for i, id_i in enumerate(id):
    #         for m, n in zip(y_perm_i[id_i], y_i[i]):
    #             assert m == n
    #

if __name__ == "__main__":
    # test_permute_tensor_batched()
    test_permute_tensor_batched_2d()

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


from experiments.jaccard_metric.utils import serialize_sq_matrix, print_vector
from experiments.jaccard_metric.k_means import k_means
import numpy as np

def helper(psis, apis, num_centroids=10):
    print('Kmeans in progress .. ', end='')
    clusters, clustered_apis = k_means(psis, apis, num_centroids=num_centroids)
    print('Done')

    print('Intra cluster Jaccards .. ', end='')
    cluster_jaccards = [get_intra_cluster_jaccards(x) for x in clustered_apis]
    print('Sorting ...', end='')

    intra_cluster_jaccards = sorted(cluster_jaccards, reverse=True)
    print_vector(intra_cluster_jaccards)
    sorted_ids = [i[0] for i in sorted(enumerate(cluster_jaccards), key=lambda x: x[1], reverse=True)]
    clusters = [clusters[i] for i in sorted_ids]
    clustered_apis = [clustered_apis[i] for i in sorted_ids]
    print('Done')

    print('Inter cluster Jaccards .. ', end='')
    num_clusters = len(clustered_apis)
    jac_matrix = np.zeros((num_clusters, num_clusters))
    for j, clustered_apis_j in enumerate(clustered_apis):
        print(j, end=',')
        for k, clustered_apis_k in enumerate(clustered_apis):
            if k > j:
                jac = get_inter_cluster_jaccards(clustered_apis_j, clustered_apis_k)
                jac_matrix[j][k] = jac
            elif k == j:
                jac = get_intra_cluster_jaccards(clustered_apis_j)
                jac_matrix[j][k] = jac
            else:
                jac_matrix[j][k] = jac_matrix[k][j]
    print('Done')
    print(serialize_sq_matrix(jac_matrix))
    return jac_matrix, intra_cluster_jaccards


def get_intra_cluster_jaccards(clustered_apis_k):
    dis_i = 0.
    count = 0.001
    for i, api_i in enumerate(clustered_apis_k):
        for j, api_j in enumerate(clustered_apis_k):
            if j <= i:
                continue
            dis_i += get_jaccard_distance(api_i, api_j)
            count += 1

    num_items = len(clustered_apis_k)
    return dis_i / count


def get_inter_cluster_jaccards(clustered_apis_j, clustered_apis_k):
    dis_ = 0.
    for api_i in clustered_apis_j:
        for api_j in clustered_apis_k:
            dis_ += get_jaccard_distance(api_i, api_j)

    num_items_1 = len(clustered_apis_j)
    num_items_2 = len(clustered_apis_k)

    return dis_ / (num_items_1 * num_items_2 + 0.0001)


def get_jaccard_distance(a, b):

    set_a = set()
    set_b = set()

    for item in a:
        set_a.add(item)

    for item in b:
        set_b.add(item)

    if (len(set_a) == 0) and (len(set_b) == 0):
        return 1

    distance = len(set_a & set_b) / len(set_a | set_b)
    return distance

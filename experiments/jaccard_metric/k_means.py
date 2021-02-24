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

from scipy.cluster.vq import kmeans2
import numpy as np


def k_means(psis, apis, num_centroids=10, max_cap=200):
    centroid, labels = kmeans2(np.array(psis), num_centroids)
    clusters = [[] for _ in range(num_centroids)]
    clustered_apis = [[] for _ in range(num_centroids)]
    for k, label in enumerate(labels):
        if len(clusters[label]) > max_cap:
            continue
        clusters[label].append(psis[k])
        clustered_apis[label].append(apis[k])
    return clusters, clustered_apis

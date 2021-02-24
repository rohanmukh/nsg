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

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json

def serialize_sq_matrix(jac_matrix):
    num_clusters = len(jac_matrix)
    text = '['
    for j in range(num_clusters):
        text += '['
        for k in range(num_clusters):
            if k == num_clusters - 1 and j == num_clusters - 1:
                text += str("%.3f" % jac_matrix[j][k]) + ']\n'
            elif k == num_clusters - 1:
                text += str("%.3f" % jac_matrix[j][k]) + '],\n'
            else:
                text += str("%.3f" % jac_matrix[j][k]) + ','
    text += ']'
    return text

def print_vector(jac_vector):
    print()
    for j, val in enumerate(jac_vector):
        print('Jaccard of Cluster :: ' + str(j) + ' is :: ', str(val))



def plotter(matrix, vector, name='temp'):
    num_centroids = len(vector)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation='nearest', cmap=matplotlib.cm.get_cmap('Blues'))
    cax.set_clim(0.0, max(vector))
    cbar = fig.colorbar(cax)
    cbar.set_label('Average Jaccard Similarity', size=18)
    cbar.ax.tick_params(labelsize=18)

    xticks = list(range(num_centroids))
    yticks = list(range(num_centroids))

    ax.set_xticks(xticks)
    ax.set_xticklabels([str(val + 1) if (val + 1) == 1 or (val + 1) % 5 == 0 else '' for val in xticks], fontsize=18)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(val + 1) if (val + 1) == 1 or (val + 1) % 5 == 0 else '' for val in yticks], fontsize=18)
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel('Cluster Number', fontsize=18)
    plt.ylabel('Cluster Number', fontsize=18)
    plt.savefig(name + '.png')
    with open(name + '.json', 'w') as f:
        json.dump({'jaccard_intra_cluster': vector, 'jaccard_inter_matrix': serialize_sq_matrix(matrix)}, f, indent=4)
    return
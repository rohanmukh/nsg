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

from experiments.tSNE_visualizor.get_labels import LABELS

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import os


def fitTSNEandplot(psis, labels, name):
    model = TSNE(n_jobs=24)
    psis_2d = model.fit_transform(np.array(psis))

    assert len(psis_2d) == len(labels)
    print('Doing a scatter plot')
    scatter(zip(psis_2d, labels), name)


def scatter(data, name):
    dic = {}
    for psi_2d, label in data:
        if label == 'N/A':
            continue
        if label not in dic:
            dic[label] = []
        dic[label].append(psi_2d)

    labels = list(dic.keys())
    labels.sort(key=lambda l: len(dic[l]), reverse=True)
    # for label in labels[topK:]:
    #     del dic[label]

    # labels = dic.keys()
    labels = LABELS
    colors = cm.rainbow(np.linspace(0, 1, len(dic)))

    fig = plt.figure()
    ax = plt.subplot(111)

    plotpoints = []
    for label, color in zip(labels, colors):
        x = list(map(lambda s: s[0], dic[label]))
        y = list(map(lambda s: s[1], dic[label]))
        plotpoints.append(plt.scatter(x, y, color=color))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    plt.legend(plotpoints, labels, scatterpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.055), ncol=4,
               fontsize=14, fancybox=True, shadow=True)
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
    #      fancybox=True, shadow=True, ncol=3)
    # plt.rcParams.update({'font.size': 14})
    plt.savefig(os.path.join(os.getcwd(), name), bbox_inches='tight')
    # plt.show()
